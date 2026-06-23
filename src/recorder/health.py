"""Health monitoring and logging for the recorder."""

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .sync import OrderbookSync, SyncState

if TYPE_CHECKING:
    from .notify import Notifier

logger = logging.getLogger(__name__)

# Disk-space thresholds (early warning before the disk fills up). A warning
# fires below LOW, an urgent alert below CRITICAL -- whichever (GB or %) hits
# first, so it works on both small and large disks.
DISK_LOW_GB = 10.0
DISK_LOW_PCT = 10.0
DISK_CRITICAL_GB = 3.0
DISK_CRITICAL_PCT = 5.0

# Number of consecutive health cycles with a near-full queue AND no progress
# before we treat the writer as stalled (blocked, but not crashed).
STALL_CYCLES = 3

# Consecutive cycles with the orderbook LIVE and the trade WS connected but no
# new trades recorded, before alerting. At 10s/cycle this is ~5 min -- well
# below any real BTCUSDT trade gap, but enough to avoid false positives.
# Catches a silent trade-stream outage (e.g. the aggTrade stream delivering
# nothing) that snapshot/queue/disk monitoring would never notice.
TRADE_STALL_CYCLES = 30


@dataclass
class HealthStats:
    """Aggregated health statistics."""

    # Sync
    sync_state: SyncState = SyncState.DISCONNECTED
    resync_count: int = 0
    last_update_id: int = 0

    # WebSocket
    depth_reconnects: int = 0
    trade_reconnects: int = 0
    depth_connected: bool = False
    trade_connected: bool = False

    # Snapshots
    snapshots_taken: int = 0
    snapshots_written: int = 0
    snapshot_queue_size: int = 0
    snapshot_queue_pct: float = 0.0

    # Trades
    trades_written: int = 0
    trade_queue_size: int = 0
    trade_queue_pct: float = 0.0

    # Timing
    uptime_s: float = 0.0
    last_snapshot_ts: int = 0


class HealthMonitor:
    """
    Monitors and logs recorder health metrics.
    
    Periodically logs:
    - Sync state and resync count
    - WebSocket connection status
    - Queue fill levels
    - Throughput stats
    """

    def __init__(
        self,
        log_interval_s: float = 10.0,
    ):
        self.log_interval_s = log_interval_s
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_time: float = 0

        # References to components (set after construction)
        self.sync: Optional[OrderbookSync] = None
        self.snapshot_writer: Optional["SnapshotWriter"] = None
        self.trade_writer: Optional["TradeWriter"] = None
        self.depth_client: Optional["WSDepthClient"] = None
        self.trade_client: Optional["WSTradeClient"] = None
        self.scheduler: Optional["SnapshotScheduler"] = None
        self.notifier: Optional["Notifier"] = None
        self.data_root: Optional[Path] = None

        # Stall detection state
        self._prev_snapshots_written = 0
        self._snapshot_stall_cycles = 0
        self._prev_trades_written = 0
        self._trade_stall_cycles = 0

    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._task = asyncio.create_task(self._run())
        logger.info("Health monitor started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    def get_stats(self) -> HealthStats:
        """Gather current health statistics."""
        stats = HealthStats()

        stats.uptime_s = time.monotonic() - self._start_time

        if self.sync:
            stats.sync_state = self.sync.state
            stats.resync_count = self.sync.stats.resync_count
            stats.last_update_id = self.sync.stats.last_update_id

        if self.depth_client:
            stats.depth_reconnects = self.depth_client.reconnect_count
            stats.depth_connected = self.depth_client.is_connected

        if self.trade_client:
            stats.trade_reconnects = self.trade_client.reconnect_count
            stats.trade_connected = self.trade_client.is_connected

        if self.scheduler:
            stats.snapshots_taken = self.scheduler.snapshots_taken

        if self.snapshot_writer:
            stats.snapshots_written = self.snapshot_writer.snapshots_written
            stats.snapshot_queue_size = self.snapshot_writer.queue_size
            stats.snapshot_queue_pct = self.snapshot_writer.queue_fill_pct

        if self.trade_writer:
            stats.trades_written = self.trade_writer.trades_written
            stats.trade_queue_size = self.trade_writer.queue_size
            stats.trade_queue_pct = self.trade_writer.queue_fill_pct

        return stats

    async def _run(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.log_interval_s)
                self._log_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Health monitor error: {e}")

    def _log_health(self) -> None:
        """Log current health status."""
        stats = self.get_stats()

        logger.info(
            f"HEALTH | "
            f"state={stats.sync_state.name} | "
            f"uptime={stats.uptime_s:.0f}s | "
            f"resyncs={stats.resync_count} | "
            f"snaps={stats.snapshots_taken}/{stats.snapshots_written} | "
            f"snap_q={stats.snapshot_queue_size} ({stats.snapshot_queue_pct:.1f}%) | "
            f"trades={stats.trades_written} | "
            f"trade_q={stats.trade_queue_size} ({stats.trade_queue_pct:.1f}%) | "
            f"ws_depth={'OK' if stats.depth_connected else 'DOWN'} | "
            f"ws_trade={'OK' if stats.trade_connected else 'DOWN'}"
        )

        # Warn on high queue fill
        if stats.snapshot_queue_pct > 50:
            logger.warning(f"Snapshot queue fill high: {stats.snapshot_queue_pct:.1f}%")
        if stats.trade_queue_pct > 50:
            logger.warning(f"Trade queue fill high: {stats.trade_queue_pct:.1f}%")

        if self.notifier:
            if stats.snapshot_queue_pct > 80:
                asyncio.create_task(
                    self.notifier.queue_pressure("snapshot", stats.snapshot_queue_pct)
                )
            if stats.trade_queue_pct > 80:
                asyncio.create_task(
                    self.notifier.queue_pressure("trade", stats.trade_queue_pct)
                )

        self._check_disk_space()
        self._check_writer_stall(stats)
        self._check_trade_flow(stats)

    def _check_disk_space(self) -> None:
        """Warn early if the data disk is running low (before writes fail)."""
        if not self.notifier or self.data_root is None:
            return
        try:
            usage = shutil.disk_usage(self.data_root)
        except OSError as e:
            logger.warning(f"Could not check disk usage for {self.data_root}: {e}")
            return

        free_gb = usage.free / (1024 ** 3)
        free_pct = (usage.free / usage.total) * 100 if usage.total else 0.0

        if free_gb < DISK_CRITICAL_GB or free_pct < DISK_CRITICAL_PCT:
            logger.error(
                f"Disk space CRITICAL: {free_gb:.1f} GB ({free_pct:.1f}%) free "
                f"on {self.data_root}"
            )
            asyncio.create_task(
                self.notifier.disk_space_critical(free_gb, free_pct, str(self.data_root))
            )
        elif free_gb < DISK_LOW_GB or free_pct < DISK_LOW_PCT:
            logger.warning(
                f"Disk space low: {free_gb:.1f} GB ({free_pct:.1f}%) free "
                f"on {self.data_root}"
            )
            asyncio.create_task(
                self.notifier.disk_space_low(free_gb, free_pct, str(self.data_root))
            )

    def _check_writer_stall(self, stats: HealthStats) -> None:
        """Detect a writer that is alive but blocked (queue full, no progress)."""
        if not self.notifier:
            return
        if stats.snapshot_queue_pct >= 95 and (
            stats.snapshots_written == self._prev_snapshots_written
        ):
            self._snapshot_stall_cycles += 1
        else:
            self._snapshot_stall_cycles = 0
        self._prev_snapshots_written = stats.snapshots_written

        if self._snapshot_stall_cycles >= STALL_CYCLES:
            logger.error(
                f"Snapshot writer appears STALLED: queue at "
                f"{stats.snapshot_queue_pct:.0f}% with no progress for "
                f"{self._snapshot_stall_cycles} cycles"
            )
            asyncio.create_task(
                self.notifier.writer_stalled("snapshot", stats.snapshot_queue_pct)
            )

    def _check_trade_flow(self, stats: HealthStats) -> None:
        """Detect a silent trade-stream outage (LIVE + connected but no trades)."""
        if not self.notifier:
            return
        # Only meaningful when we should actually be receiving trades.
        if stats.sync_state != SyncState.LIVE or not stats.trade_connected:
            self._trade_stall_cycles = 0
            self._prev_trades_written = stats.trades_written
            return

        if stats.trades_written == self._prev_trades_written:
            self._trade_stall_cycles += 1
        else:
            self._trade_stall_cycles = 0
        self._prev_trades_written = stats.trades_written

        if self._trade_stall_cycles >= TRADE_STALL_CYCLES:
            minutes = (self._trade_stall_cycles * self.log_interval_s) / 60.0
            logger.error(
                f"Trade stream appears STALLED: LIVE and ws_trade connected but "
                f"no trades recorded for {minutes:.0f} min"
            )
            asyncio.create_task(self.notifier.trades_stalled(minutes))


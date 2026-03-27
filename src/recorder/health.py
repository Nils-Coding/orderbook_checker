"""Health monitoring and logging for the recorder."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from .sync import OrderbookSync, SyncState

if TYPE_CHECKING:
    from .notify import Notifier

logger = logging.getLogger(__name__)


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


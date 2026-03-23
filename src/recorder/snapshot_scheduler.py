"""Snapshot scheduler for 100ms orderbook captures."""

import asyncio
import logging
import time
from typing import Callable, Awaitable, Optional, TYPE_CHECKING

from .orderbook import Orderbook, OrderbookSnapshot

if TYPE_CHECKING:
    from .sync import OrderbookSync

logger = logging.getLogger(__name__)


class SnapshotScheduler:
    """
    Schedules orderbook snapshots at fixed intervals.
    
    - Triggers snapshot every 100ms (configurable)
    - Uses Binance event timestamp (E field) for accurate timing
    - Does NOT block on I/O
    - Sends snapshots to callback (which enqueues to writer)
    """

    def __init__(
        self,
        orderbook: Orderbook,
        interval_ms: int = 100,
        on_snapshot: Optional[Callable[[OrderbookSnapshot], Awaitable[bool]]] = None,
    ):
        self.orderbook = orderbook
        self.interval_ms = interval_ms
        self.on_snapshot = on_snapshot
        self.sync: Optional["OrderbookSync"] = None  # Set by Recorder after init

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._enabled = False  # Only capture when sync is LIVE

        # Stats
        self.snapshots_taken = 0
        self.snapshots_dropped = 0

    @property
    def interval_s(self) -> float:
        return self.interval_ms / 1000.0

    def enable(self) -> None:
        """Enable snapshot capture (called when sync goes LIVE)."""
        self._enabled = True
        logger.info("Snapshot scheduler enabled")

    def disable(self) -> None:
        """Disable snapshot capture (called on resync)."""
        self._enabled = False
        logger.info("Snapshot scheduler disabled")

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"Snapshot scheduler started ({self.interval_ms}ms interval)")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Snapshot scheduler stopped")

    async def _run(self) -> None:
        """Main scheduler loop."""
        next_tick = time.monotonic()

        while self._running:
            now = time.monotonic()

            if now >= next_tick:
                if self._enabled:
                    await self._take_snapshot()

                # Schedule next tick
                next_tick += self.interval_s
                # If we're behind, catch up but don't flood
                if next_tick < now:
                    skipped = int((now - next_tick) / self.interval_s)
                    next_tick = now + self.interval_s
                    if skipped > 0:
                        logger.warning(f"Scheduler behind, skipped {skipped} ticks")
                        self.snapshots_dropped += skipped

            # Sleep until next tick
            sleep_time = max(0, next_tick - time.monotonic())
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _take_snapshot(self) -> None:
        """Take a snapshot and send to callback."""
        # Use Binance event timestamp if available, otherwise fall back to local time
        event_ts_ns = self.sync.last_event_ts_ns if self.sync else 0
        snapshot = self.orderbook.get_snapshot(event_ts_ns=event_ts_ns)

        if self.on_snapshot:
            # Callback returns False if queue is full (backpressure)
            success = await self.on_snapshot(snapshot)
            if success:
                self.snapshots_taken += 1
            else:
                self.snapshots_dropped += 1
        else:
            self.snapshots_taken += 1


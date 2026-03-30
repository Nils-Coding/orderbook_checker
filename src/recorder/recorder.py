"""Main recorder orchestrator combining all components."""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp

from .config import Config
from .orderbook import Orderbook, OrderbookSnapshot
from .sync import OrderbookSync, SyncState
from .ws_depth_client import WSDepthClient
from .ws_trade_client import WSTradeClient
from .snapshot_scheduler import SnapshotScheduler
from .writers import SnapshotWriter, TradeWriter, TradeRecord
from .health import HealthMonitor
from .notify import Notifier

logger = logging.getLogger(__name__)


class Recorder:
    """
    Main recorder orchestrator.
    
    Coordinates:
    - WebSocket clients (depth + trades)
    - Orderbook synchronization
    - Snapshot scheduling
    - Parquet writers
    - Health monitoring
    """

    def __init__(self, config: Config, symbol: str):
        self.config = config
        self.symbol = symbol

        # Core components
        self.orderbook = Orderbook(
            symbol=symbol,
            max_depth=config.book_depth,
            price_to_ticks=config.price_to_ticks,
            qty_to_lots=config.qty_to_lots,
        )

        self.sync: Optional[OrderbookSync] = None
        self.depth_client: Optional[WSDepthClient] = None
        self.trade_client: Optional[WSTradeClient] = None
        self.scheduler: Optional[SnapshotScheduler] = None
        self.snapshot_writer: Optional[SnapshotWriter] = None
        self.trade_writer: Optional[TradeWriter] = None
        self.health: Optional[HealthMonitor] = None
        self.notifier: Notifier = Notifier(
            topic=config.notify_topic,
            host_name=config.host_name,
            enabled=config.notify_enabled,
        )

        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._sync_recovery_task: Optional[asyncio.Task] = None
        self._sync_recovery_attempts = 0

    async def start(self) -> None:
        """Initialize and start all components."""
        logger.info(f"Starting recorder for {self.symbol}")

        # Create HTTP session
        self._session = aiohttp.ClientSession()

        # Initialize writers
        self.snapshot_writer = SnapshotWriter(
            data_root=self.config.data_root,
            symbol=self.symbol,
            queue_max=self.config.writer_queue_max,
            chunk_seconds=self.config.chunk_seconds,
            fail_on_backpressure=self.config.fail_on_backpressure,
        )

        self.trade_writer = TradeWriter(
            data_root=self.config.data_root,
            symbol=self.symbol,
            queue_max=self.config.writer_queue_max,
            chunk_seconds=self.config.chunk_seconds,
            fail_on_backpressure=self.config.fail_on_backpressure,
        )

        # Initialize sync
        self.sync = OrderbookSync(
            orderbook=self.orderbook,
            rest_url=self.config.binance_rest_url,
            symbol=self.symbol,
            snapshot_limit=self.config.rest_snapshot_limit,
            on_state_change=self._on_sync_state_change,
            on_live=self._on_sync_live,
        )

        # Initialize scheduler
        self.scheduler = SnapshotScheduler(
            orderbook=self.orderbook,
            interval_ms=self.config.snapshot_interval_ms,
            on_snapshot=self._on_snapshot,
        )
        self.scheduler.sync = self.sync  # Link to sync for Binance event timestamps

        # Initialize WebSocket clients
        self.depth_client = WSDepthClient(
            ws_url=self.config.binance_ws_url,
            symbol=self.symbol,
            stream_suffix=self.config.ws_depth_stream,
            on_event=self._on_depth_event,
            on_connect=self._on_depth_connect,
            on_disconnect=self._on_depth_disconnect,
        )

        self.trade_client = WSTradeClient(
            ws_url=self.config.binance_ws_url,
            symbol=self.symbol,
            stream_type=self.config.ws_trade_stream,
            on_event=self._on_trade_event,
        )

        # Initialize health monitor
        self.health = HealthMonitor(
            log_interval_s=self.config.health_log_interval_s,
        )
        self.health.sync = self.sync
        self.health.snapshot_writer = self.snapshot_writer
        self.health.trade_writer = self.trade_writer
        self.health.depth_client = self.depth_client
        self.health.trade_client = self.trade_client
        self.health.scheduler = self.scheduler
        self.health.notifier = self.notifier

        # Start all components
        self._running = True

        await self.notifier.start()
        await self.snapshot_writer.start()
        await self.trade_writer.start()
        await self.scheduler.start()
        await self.health.start()
        await self.depth_client.start()
        await self.trade_client.start()

        logger.info(f"Recorder started for {self.symbol}")
        await self.notifier.recorder_started(self.symbol)

    async def stop(self) -> None:
        """Stop all components gracefully."""
        logger.info(f"Stopping recorder for {self.symbol}")
        self._running = False
        if self._sync_recovery_task and not self._sync_recovery_task.done():
            self._sync_recovery_task.cancel()
            try:
                await self._sync_recovery_task
            except asyncio.CancelledError:
                pass

        # Stop in reverse order
        if self.depth_client:
            await self.depth_client.stop()
        if self.trade_client:
            await self.trade_client.stop()
        if self.scheduler:
            await self.scheduler.stop()
        if self.health:
            await self.health.stop()
        if self.snapshot_writer:
            await self.snapshot_writer.stop()
        if self.trade_writer:
            await self.trade_writer.stop()
        if self._session:
            await self._session.close()

        logger.info(f"Recorder stopped for {self.symbol}")
        await self.notifier.recorder_stopped(self.symbol)
        await self.notifier.stop()

    async def run_until_shutdown(self) -> int:
        """Run until shutdown signal received. Returns exit code."""
        try:
            await self._shutdown_event.wait()
            return 0
        except Exception as e:
            logger.exception(f"Recorder error: {e}")
            await self.notifier.recorder_error(self.symbol, str(e))
            return 1

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self._shutdown_event.set()

    # === Callbacks ===

    def _on_sync_state_change(self, old: SyncState, new: SyncState) -> None:
        """Handle sync state transitions."""
        if new == SyncState.RESYNCING:
            if self.scheduler:
                self.scheduler.disable()
            asyncio.create_task(
                self.notifier.sync_lost(self.symbol, f"{old.name} -> RESYNCING")
            )
        elif new == SyncState.BUFFERING:
            self._ensure_sync_recovery_task(reason=f"{old.name} -> BUFFERING")
        elif new == SyncState.LIVE:
            if self.scheduler:
                self.scheduler.enable()
            if old != SyncState.SYNCING:
                asyncio.create_task(self.notifier.sync_live(self.symbol))
            self._sync_recovery_attempts = 0

    async def _on_sync_live(self) -> None:
        """Called when sync reaches LIVE state."""
        logger.info(f"Orderbook LIVE for {self.symbol}: {self.orderbook}")

    async def _on_snapshot(self, snapshot: OrderbookSnapshot) -> bool:
        """Handle snapshot from scheduler."""
        if self.snapshot_writer:
            return await self.snapshot_writer.enqueue(snapshot)
        return True

    def _on_depth_event(self, event: dict) -> None:
        """Handle depth event from WebSocket."""
        if self.sync:
            # Check if we're in SYNCING state and need to process manually
            if self.sync.state == SyncState.SYNCING:
                # Schedule async processing
                asyncio.create_task(self.sync.process_during_sync(event))
            else:
                self.sync.on_ws_event(event)

    async def _on_depth_connect(self) -> None:
        """Handle depth WebSocket connection."""
        logger.info("Depth WebSocket connected")
        if self.sync and self._session:
            await self.sync.start()
            self._ensure_sync_recovery_task(reason="depth websocket connected")

    async def _on_depth_disconnect(self) -> None:
        """Handle depth WebSocket disconnection."""
        logger.warning("Depth WebSocket disconnected")
        if self.sync:
            await self.sync.trigger_resync("websocket disconnected")

    def _ensure_sync_recovery_task(self, reason: str) -> None:
        """Ensure one background task is trying to recover sync to LIVE."""
        if not self._running or not self.sync or not self._session:
            return
        if self.sync.state == SyncState.LIVE:
            return
        if self._sync_recovery_task and not self._sync_recovery_task.done():
            return
        logger.info(f"Starting sync recovery loop for {self.symbol} ({reason})")
        self._sync_recovery_task = asyncio.create_task(self._sync_recovery_loop())

    def _should_notify_recovery(self) -> bool:
        """Notify only after at least one real resync happened."""
        return bool(self.sync and self.sync.stats.resync_count > 0)

    async def _sync_recovery_loop(self) -> None:
        """
        Try to reach LIVE from BUFFERING/SYNCING with bounded retry backoff.

        This prevents the recorder from getting stuck in BUFFERING after
        a sequence-gap-triggered resync where no new connect callback fires.
        """
        try:
            while self._running and self.sync and self._session:
                if self.sync.state == SyncState.LIVE:
                    return

                if self.sync.state in (SyncState.DISCONNECTED, SyncState.RESYNCING):
                    await asyncio.sleep(0.5)
                    continue

                if self.sync.state in (SyncState.FETCH_SNAPSHOT, SyncState.SYNCING):
                    await asyncio.sleep(0.5)
                    continue

                # BUFFERING: fetch snapshot + attempt to sync.
                self._sync_recovery_attempts += 1
                attempt = self._sync_recovery_attempts
                logger.warning(
                    f"Sync recovery attempt {attempt} for {self.symbol} (state=BUFFERING)"
                )

                # Buffer a few websocket events before REST snapshot fetch.
                await asyncio.sleep(0.5)
                success = await self.sync.fetch_and_sync(self._session)
                if success and self.sync.state == SyncState.LIVE:
                    if attempt > 1:
                        logger.info(
                            f"Sync recovery succeeded for {self.symbol} after {attempt} attempts"
                        )
                        if self._should_notify_recovery():
                            asyncio.create_task(
                                self.notifier.sync_recovered(self.symbol, attempt)
                            )
                    return

                delay_s = min(30.0, float(2 ** min(6, attempt - 1)))
                logger.warning(
                    f"Sync recovery attempt {attempt} did not reach LIVE for {self.symbol}; "
                    f"retry in {delay_s:.1f}s"
                )
                if self._should_notify_recovery():
                    asyncio.create_task(
                        self.notifier.sync_recovery_retry(self.symbol, attempt, delay_s)
                    )
                await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            logger.debug(f"Sync recovery loop cancelled for {self.symbol}")
            raise
        except Exception as e:
            logger.exception(f"Sync recovery loop error for {self.symbol}: {e}")

    def _on_trade_event(self, event: dict) -> None:
        """Handle trade event from WebSocket."""
        if not self.trade_writer or not self._running:
            return

        # Only record trades when orderbook is live
        if self.sync and self.sync.state != SyncState.LIVE:
            return

        try:
            # Parse trade event - supports both @trade and @aggTrade formats
            # @trade format: {t, s, p, q, T, m, ...}
            # @aggTrade format: {a, s, p, q, f, l, T, m}
            ts_ms = event.get("T", 0)  # Trade time in ms
            price = float(event.get("p", 0))
            qty = float(event.get("q", 0))
            is_buyer_maker = event.get("m", False)
            
            # aggTrade specific fields (defaults to 0 for @trade stream)
            agg_trade_id = event.get("a", 0)      # Aggregate trade ID
            first_trade_id = event.get("f", 0)   # First trade ID in aggregate
            last_trade_id = event.get("l", 0)    # Last trade ID in aggregate

            trade = TradeRecord(
                ts_ns=ts_ms * 1_000_000,  # Convert ms to ns
                symbol=self.symbol,
                price_ticks=self.config.price_to_ticks(price),
                qty_lots=self.config.qty_to_lots(qty),
                is_buyer_maker=is_buyer_maker,
                agg_trade_id=agg_trade_id,
                first_trade_id=first_trade_id,
                last_trade_id=last_trade_id,
            )

            # Enqueue trade (non-blocking)
            asyncio.create_task(self.trade_writer.enqueue(trade))

        except Exception as e:
            logger.warning(f"Failed to process trade: {e}")


async def run_recorder(config: Config) -> int:
    """
    Run the recorder for all configured symbols.
    
    Returns exit code (0 = success, non-zero = error).
    """
    recorders: list[Recorder] = []
    exit_code = 0

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        for r in recorders:
            r.request_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Create and start recorders for each symbol
        for symbol in config.symbols:
            recorder = Recorder(config, symbol)
            recorders.append(recorder)
            await recorder.start()

        # Wait for all recorders
        results = await asyncio.gather(
            *[r.run_until_shutdown() for r in recorders],
            return_exceptions=True,
        )

        # Check for errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Recorder {config.symbols[i]} failed: {result}")
                exit_code = 1
            elif result != 0:
                exit_code = result

    except Exception as e:
        logger.exception(f"Recorder error: {e}")
        exit_code = 1
        for r in recorders:
            await r.notifier.recorder_error(
                r.symbol, f"Fatal: {e}"
            )

    finally:
        # Stop all recorders
        for recorder in recorders:
            try:
                await recorder.stop()
            except Exception as e:
                logger.warning(f"Error stopping recorder: {e}")

    return exit_code


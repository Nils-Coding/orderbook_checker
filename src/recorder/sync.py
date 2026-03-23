"""Orderbook synchronization state machine following Binance protocol."""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable, Awaitable

import aiohttp

from .orderbook import Orderbook

logger = logging.getLogger(__name__)


class SyncState(Enum):
    """Synchronization state machine states."""

    DISCONNECTED = auto()
    BUFFERING = auto()  # WS connected, buffering events
    FETCH_SNAPSHOT = auto()  # Fetching REST snapshot
    SYNCING = auto()  # Looking for first valid event
    LIVE = auto()  # Normal operation
    RESYNCING = auto()  # Error recovery


@dataclass
class SyncStats:
    """Statistics for sync operations."""

    resync_count: int = 0
    events_processed: int = 0
    events_buffered: int = 0
    events_dropped: int = 0
    last_update_id: int = 0
    state: SyncState = SyncState.DISCONNECTED


class OrderbookSync:
    """
    Manages orderbook synchronization with Binance.
    
    State Machine:
    1. BUFFERING: WS connected, buffer incoming events
    2. FETCH_SNAPSHOT: Request REST snapshot
    3. SYNCING: Find first valid event (U <= lastUpdateId <= u)
    4. LIVE: Normal operation, check pu == prev.u
    5. RESYNCING: On error, restart from BUFFERING
    
    Rules (Binance USD-M Futures):
    - First event after snapshot: U <= lastUpdateId <= u
    - Subsequent events: pu == previous.u
    - qty = 0 means delete level
    - qty > 0 means absolute quantity (replace)
    """

    def __init__(
        self,
        orderbook: Orderbook,
        rest_url: str,
        symbol: str,
        snapshot_limit: int = 1000,
        buffer_size: int = 1000,
        on_state_change: Optional[Callable[[SyncState, SyncState], None]] = None,
        on_live: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.orderbook = orderbook
        self.rest_url = rest_url
        self.symbol = symbol
        self.snapshot_limit = snapshot_limit
        self.buffer_size = buffer_size
        self.on_state_change = on_state_change
        self.on_live = on_live

        self._state = SyncState.DISCONNECTED
        self._buffer: deque[dict] = deque(maxlen=buffer_size)
        self._snapshot_last_update_id: int = 0
        self._prev_u: int = 0
        self._first_event_applied: bool = False
        self._last_event_ts_ns: int = 0  # Binance event timestamp (E field) in nanoseconds

        self.stats = SyncStats()

    @property
    def state(self) -> SyncState:
        return self._state

    @property
    def last_event_ts_ns(self) -> int:
        """Last Binance event timestamp in nanoseconds (from 'E' field)."""
        return self._last_event_ts_ns

    def _set_state(self, new_state: SyncState) -> None:
        """Transition to new state with logging and callback."""
        if new_state != self._state:
            old_state = self._state
            logger.info(f"Sync state transition: {old_state.name} -> {new_state.name}")
            self._state = new_state
            self.stats.state = new_state
            if self.on_state_change:
                self.on_state_change(old_state, new_state)

    async def start(self) -> None:
        """Start synchronization process."""
        self._set_state(SyncState.BUFFERING)
        self._buffer.clear()
        self._first_event_applied = False

    async def trigger_resync(self, reason: str = "unknown") -> None:
        """Trigger resynchronization."""
        logger.warning(f"Resync triggered: {reason}")
        self.stats.resync_count += 1
        self.orderbook.resync_epoch += 1
        self._set_state(SyncState.RESYNCING)
        self._buffer.clear()
        self._first_event_applied = False
        # Transition back to buffering
        self._set_state(SyncState.BUFFERING)

    def on_ws_event(self, event: dict) -> None:
        """
        Handle incoming WebSocket depth event.
        
        Called from WS client, must be non-blocking.
        """
        if self._state == SyncState.DISCONNECTED:
            return

        if self._state in (SyncState.BUFFERING, SyncState.FETCH_SNAPSHOT, SyncState.SYNCING):
            # Buffer events during sync process
            self._buffer.append(event)
            self.stats.events_buffered += 1
            return

        if self._state == SyncState.LIVE:
            # Normal processing
            self._process_live_event(event)

    def _process_live_event(self, event: dict) -> bool:
        """Process event in LIVE state. Returns True if successful."""
        U = event.get("U", 0)  # First update ID in event
        u = event.get("u", 0)  # Last update ID in event
        pu = event.get("pu", 0)  # Previous update ID
        E = event.get("E", 0)  # Event timestamp in milliseconds (Binance server time)

        # Check sequence: pu must equal previous u
        if pu != self._prev_u:
            logger.error(
                f"Sequence gap: pu={pu}, expected={self._prev_u}, U={U}, u={u}"
            )
            # Schedule resync (cannot await here)
            asyncio.create_task(self.trigger_resync(f"sequence gap: pu={pu} != prev_u={self._prev_u}"))
            return False

        # Apply diff
        self.orderbook.apply_diff(event)
        self._prev_u = u
        self._last_event_ts_ns = E * 1_000_000  # Convert ms to ns
        self.stats.events_processed += 1
        self.stats.last_update_id = u
        return True

    async def fetch_and_sync(self, session: aiohttp.ClientSession) -> bool:
        """
        Fetch REST snapshot and sync with buffered events.
        
        Returns True if sync successful, False if resync needed.
        """
        self._set_state(SyncState.FETCH_SNAPSHOT)

        # Fetch REST snapshot
        url = f"{self.rest_url}/fapi/v1/depth"
        params = {"symbol": self.symbol, "limit": self.snapshot_limit}

        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"REST snapshot failed: {resp.status}")
                    await self.trigger_resync(f"REST error: {resp.status}")
                    return False
                snapshot = await resp.json()
        except Exception as e:
            logger.exception(f"REST snapshot error: {e}")
            await self.trigger_resync(f"REST exception: {e}")
            return False

        last_update_id = snapshot.get("lastUpdateId", 0)
        logger.info(f"REST snapshot received: lastUpdateId={last_update_id}")

        # Apply snapshot to orderbook
        self.orderbook.apply_snapshot(snapshot)
        self._snapshot_last_update_id = last_update_id

        # Transition to SYNCING
        self._set_state(SyncState.SYNCING)

        # Process buffered events to find sync point
        found_sync = False
        events_to_process = list(self._buffer)
        self._buffer.clear()

        for event in events_to_process:
            U = event.get("U", 0)
            u = event.get("u", 0)

            # Rule: discard events with u < lastUpdateId
            if u < last_update_id:
                self.stats.events_dropped += 1
                continue

            if not found_sync:
                # First valid event: U <= lastUpdateId <= u
                if U <= last_update_id <= u:
                    logger.info(
                        f"Sync point found: U={U}, lastUpdateId={last_update_id}, u={u}"
                    )
                    found_sync = True
                    self._first_event_applied = True
                    self.orderbook.apply_diff(event)
                    self._prev_u = u
                    self.stats.events_processed += 1
                else:
                    # Event doesn't contain sync point, skip
                    self.stats.events_dropped += 1
                    continue
            else:
                # Subsequent events: check pu == prev_u
                pu = event.get("pu", 0)
                if pu != self._prev_u:
                    logger.error(f"Sequence gap in buffer: pu={pu}, expected={self._prev_u}")
                    await self.trigger_resync(f"buffer sequence gap")
                    return False

                self.orderbook.apply_diff(event)
                self._prev_u = u
                self.stats.events_processed += 1

        if not found_sync:
            logger.warning("No sync point found in buffer, waiting for more events")
            # Stay in SYNCING, will process new events
            return True

        # Successfully synced
        self._set_state(SyncState.LIVE)
        self.stats.last_update_id = self._prev_u

        if self.on_live:
            await self.on_live()

        return True

    async def process_during_sync(self, event: dict) -> bool:
        """
        Process event while in SYNCING state (no sync point found yet).
        
        Returns True if sync point found, False otherwise.
        """
        if self._state != SyncState.SYNCING:
            return False

        U = event.get("U", 0)
        u = event.get("u", 0)

        # Rule: discard events with u < lastUpdateId
        if u < self._snapshot_last_update_id:
            self.stats.events_dropped += 1
            return False

        if not self._first_event_applied:
            # First valid event: U <= lastUpdateId <= u
            if U <= self._snapshot_last_update_id <= u:
                logger.info(
                    f"Sync point found (late): U={U}, lastUpdateId={self._snapshot_last_update_id}, u={u}"
                )
                self._first_event_applied = True
                self.orderbook.apply_diff(event)
                self._prev_u = u
                self.stats.events_processed += 1

                # Go LIVE
                self._set_state(SyncState.LIVE)
                self.stats.last_update_id = self._prev_u

                if self.on_live:
                    await self.on_live()

                return True
            else:
                self.stats.events_dropped += 1
                return False
        else:
            # Should not reach here if first event was applied
            return False


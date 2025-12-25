"""WebSocket client for Binance USD-M Futures depth stream."""

import asyncio
import json
import logging
from typing import Callable, Optional, Awaitable

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class WSDepthClient:
    """
    WebSocket client for Binance depth diff stream.
    
    Connects to <symbol>@depth@100ms stream and forwards events
    to the provided callback.
    """

    def __init__(
        self,
        ws_url: str,
        symbol: str,
        stream_suffix: str = "depth@100ms",
        on_event: Optional[Callable[[dict], None]] = None,
        on_connect: Optional[Callable[[], Awaitable[None]]] = None,
        on_disconnect: Optional[Callable[[], Awaitable[None]]] = None,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
    ):
        self.ws_url = ws_url
        self.symbol = symbol.lower()
        self.stream_suffix = stream_suffix
        self.on_event = on_event
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self._running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._task: Optional[asyncio.Task] = None
        self._reconnect_count = 0

    @property
    def stream_name(self) -> str:
        """Full stream name for URL."""
        return f"{self.symbol}@{self.stream_suffix}"

    @property
    def full_url(self) -> str:
        """Full WebSocket URL."""
        return f"{self.ws_url}/{self.stream_name}"

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def is_connected(self) -> bool:
        if self._ws is None:
            return False
        # websockets 12+ uses .state instead of .open
        try:
            from websockets.protocol import State
            return self._ws.state == State.OPEN
        except (ImportError, AttributeError):
            # Fallback for older versions
            return getattr(self._ws, 'open', False)

    async def start(self) -> None:
        """Start the WebSocket client."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"WSDepthClient started for {self.stream_name}")

    async def stop(self) -> None:
        """Stop the WebSocket client."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"WSDepthClient stopped for {self.stream_name}")

    async def _run(self) -> None:
        """Main run loop with reconnection logic."""
        current_delay = self.reconnect_delay

        while self._running:
            try:
                await self._connect_and_listen()
                # Reset delay on successful connection
                current_delay = self.reconnect_delay
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                if self.on_disconnect:
                    await self.on_disconnect()
            except Exception as e:
                logger.exception(f"WebSocket error: {e}")
                if self.on_disconnect:
                    await self.on_disconnect()

            if self._running:
                self._reconnect_count += 1
                logger.info(f"Reconnecting in {current_delay}s (attempt {self._reconnect_count})")
                await asyncio.sleep(current_delay)
                current_delay = min(current_delay * 2, self.max_reconnect_delay)

    async def _connect_and_listen(self) -> None:
        """Connect and listen for messages."""
        logger.info(f"Connecting to {self.full_url}")

        async with websockets.connect(
            self.full_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            logger.info(f"Connected to {self.stream_name}")

            if self.on_connect:
                await self.on_connect()

            async for message in ws:
                if not self._running:
                    break
                try:
                    event = json.loads(message)
                    if self.on_event:
                        self.on_event(event)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON: {e}")

        self._ws = None


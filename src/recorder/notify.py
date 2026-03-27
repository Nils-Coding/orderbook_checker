"""Push notifications via ntfy.sh for recorder state changes."""

import asyncio
import logging
import time
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

NTFY_BASE_URL = "https://ntfy.sh"


class Notifier:
    """
    Fire-and-forget push notifications via ntfy.sh.

    All errors are caught internally -- a notification failure
    must never crash or block the recorder.
    """

    def __init__(self, topic: str, host_name: str, enabled: bool = True):
        self.topic = topic
        self.host_name = host_name
        self.enabled = enabled
        self._session: Optional[aiohttp.ClientSession] = None
        self._cooldowns: dict[str, float] = {}

    async def start(self) -> None:
        if self.enabled:
            self._session = aiohttp.ClientSession()

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _check_cooldown(self, key: str, cooldown_s: float) -> bool:
        """Return True if the cooldown has expired (i.e. sending is allowed)."""
        now = time.monotonic()
        last = self._cooldowns.get(key, 0.0)
        if now - last < cooldown_s:
            return False
        self._cooldowns[key] = now
        return True

    async def send(
        self,
        title: str,
        message: str,
        priority: str = "default",
        tags: Optional[list[str]] = None,
        cooldown_key: Optional[str] = None,
        cooldown_s: float = 0,
    ) -> None:
        """
        Send a push notification. Never raises.

        Args:
            title: Notification title.
            message: Notification body.
            priority: ntfy priority (min, low, default, high, urgent).
            tags: ntfy tag/emoji names (e.g. ["warning", "rocket"]).
            cooldown_key: If set, deduplicate with this key.
            cooldown_s: Minimum seconds between messages with the same cooldown_key.
        """
        if not self.enabled or not self._session:
            return

        if cooldown_key and not self._check_cooldown(cooldown_key, cooldown_s):
            return

        prefixed_title = f"[{self.host_name}] {title}"

        headers: dict[str, str] = {
            "Title": prefixed_title,
            "Priority": priority,
        }
        if tags:
            headers["Tags"] = ",".join(tags)

        try:
            async with self._session.post(
                f"{NTFY_BASE_URL}/{self.topic}",
                data=message.encode("utf-8"),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status >= 400:
                    logger.warning(f"ntfy returned {resp.status}: {await resp.text()}")
        except Exception as e:
            logger.debug(f"ntfy send failed (non-critical): {e}")

    # --- Convenience methods for recorder events ---

    async def recorder_started(self, symbol: str) -> None:
        await self.send(
            title="Recorder started",
            message=f"Recording {symbol} -- orderbook & trades",
            tags=["rocket"],
        )

    async def recorder_stopped(self, symbol: str, reason: str = "shutdown") -> None:
        await self.send(
            title="Recorder stopped",
            message=f"{symbol} stopped: {reason}",
            priority="urgent",
            tags=["rotating_light"],
        )

    async def recorder_error(self, symbol: str, error: str) -> None:
        await self.send(
            title="Recorder ERROR",
            message=f"{symbol}: {error}",
            priority="urgent",
            tags=["rotating_light"],
        )

    async def sync_lost(self, symbol: str, reason: str) -> None:
        await self.send(
            title="Sync lost -- resyncing",
            message=f"{symbol}: {reason}",
            priority="high",
            tags=["warning"],
            cooldown_key="sync_lost",
            cooldown_s=60,
        )

    async def sync_live(self, symbol: str) -> None:
        await self.send(
            title="Sync LIVE",
            message=f"{symbol} orderbook synchronised",
            tags=["white_check_mark"],
        )

    async def queue_pressure(self, queue_name: str, fill_pct: float) -> None:
        await self.send(
            title=f"Queue pressure: {queue_name}",
            message=f"{queue_name} queue at {fill_pct:.1f}%",
            priority="high",
            tags=["warning"],
            cooldown_key=f"queue_{queue_name}",
            cooldown_s=300,
        )

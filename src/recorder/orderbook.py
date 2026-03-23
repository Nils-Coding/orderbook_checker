"""Orderbook data structure with Top-1000 management."""

from dataclasses import dataclass, field
from typing import Optional
import time

from sortedcontainers import SortedDict


@dataclass
class OrderbookSnapshot:
    """Immutable snapshot of orderbook state."""

    ts_ns: int
    symbol: str
    u: int  # Last update ID
    best_bid_ticks: int
    best_ask_ticks: int
    bids_price_delta: list[int]  # Delta from best_bid (negative = lower price)
    bids_qty_lots: list[int]
    asks_price_delta: list[int]  # Delta from best_ask (positive = higher price)
    asks_qty_lots: list[int]
    resync_epoch: int = 0


class Orderbook:
    """
    Local orderbook maintaining Top-1000 levels for bids and asks.
    
    - Bids: sorted descending by price (highest first)
    - Asks: sorted ascending by price (lowest first)
    - After each apply, trim to top 1000 levels
    """

    def __init__(
        self,
        symbol: str,
        max_depth: int = 1000,
        price_to_ticks: Optional[callable] = None,
        qty_to_lots: Optional[callable] = None,
    ):
        self.symbol = symbol
        self.max_depth = max_depth
        self._price_to_ticks = price_to_ticks or (lambda x: int(round(x * 10)))
        self._qty_to_lots = qty_to_lots or (lambda x: int(round(x * 1000)))

        # SortedDict with negated keys for bids (highest first)
        # Key: price_ticks (negated for bids), Value: qty_lots
        self._bids: SortedDict[int, int] = SortedDict()  # -price -> qty
        self._asks: SortedDict[int, int] = SortedDict()  # price -> qty

        self.last_update_id: int = 0
        self.resync_epoch: int = 0

    def clear(self) -> None:
        """Clear orderbook state."""
        self._bids.clear()
        self._asks.clear()
        self.last_update_id = 0

    def apply_snapshot(self, snapshot_data: dict) -> None:
        """
        Apply REST snapshot to orderbook.
        
        Args:
            snapshot_data: REST response with 'lastUpdateId', 'bids', 'asks'
        """
        self.clear()
        self.last_update_id = snapshot_data["lastUpdateId"]

        for price_str, qty_str in snapshot_data.get("bids", []):
            price_ticks = self._price_to_ticks(float(price_str))
            qty_lots = self._qty_to_lots(float(qty_str))
            if qty_lots > 0:
                self._bids[-price_ticks] = qty_lots

        for price_str, qty_str in snapshot_data.get("asks", []):
            price_ticks = self._price_to_ticks(float(price_str))
            qty_lots = self._qty_to_lots(float(qty_str))
            if qty_lots > 0:
                self._asks[price_ticks] = qty_lots

        self._trim()

    def apply_diff(self, event: dict) -> bool:
        """
        Apply diff event to orderbook.
        
        Args:
            event: WS depth event with 'U', 'u', 'pu', 'b', 'a'
            
        Returns:
            True if applied successfully, False if sequence invalid
        """
        # Apply bid updates
        for price_str, qty_str in event.get("b", []):
            price_ticks = self._price_to_ticks(float(price_str))
            qty_lots = self._qty_to_lots(float(qty_str))
            if qty_lots == 0:
                self._bids.pop(-price_ticks, None)
            else:
                self._bids[-price_ticks] = qty_lots

        # Apply ask updates
        for price_str, qty_str in event.get("a", []):
            price_ticks = self._price_to_ticks(float(price_str))
            qty_lots = self._qty_to_lots(float(qty_str))
            if qty_lots == 0:
                self._asks.pop(price_ticks, None)
            else:
                self._asks[price_ticks] = qty_lots

        self.last_update_id = event["u"]
        self._trim()
        return True

    def _trim(self) -> None:
        """Trim orderbook to max_depth levels on each side."""
        # Trim bids (keep first max_depth entries = highest prices)
        # SortedDict is sorted ascending, so we remove from the end (lowest bids)
        while len(self._bids) > self.max_depth:
            self._bids.popitem()  # Remove last item (lowest bid, since keys are negated)

        # Trim asks (keep first max_depth entries = lowest prices)
        # SortedDict is sorted ascending, so we remove from the end (highest asks)
        while len(self._asks) > self.max_depth:
            self._asks.popitem()  # Remove last item (highest ask)

    def get_snapshot(self, event_ts_ns: int = 0) -> OrderbookSnapshot:
        """
        Get current orderbook state as snapshot.
        
        Args:
            event_ts_ns: Binance event timestamp in nanoseconds. If 0, uses local time.
                         This should come from the 'E' field of Depth WebSocket events.
        
        Returns exactly 1000 levels per side, with padding if needed.
        Prices stored as delta from best bid/ask.
        """
        ts_ns = event_ts_ns if event_ts_ns > 0 else time.time_ns()

        # Get best prices
        best_bid_ticks = -self._bids.peekitem(0)[0] if self._bids else 0
        best_ask_ticks = self._asks.peekitem(0)[0] if self._asks else 0

        # Build bid arrays (delta from best_bid, negative = lower price)
        bids_price_delta = []
        bids_qty_lots = []
        for neg_price, qty in self._bids.items():
            price_ticks = -neg_price
            delta = price_ticks - best_bid_ticks  # Will be 0 or negative
            bids_price_delta.append(delta)
            bids_qty_lots.append(qty)

        # Build ask arrays (delta from best_ask, positive = higher price)
        asks_price_delta = []
        asks_qty_lots = []
        for price_ticks, qty in self._asks.items():
            delta = price_ticks - best_ask_ticks  # Will be 0 or positive
            asks_price_delta.append(delta)
            asks_qty_lots.append(qty)

        # Pad to exactly max_depth levels
        while len(bids_price_delta) < self.max_depth:
            bids_price_delta.append(0)
            bids_qty_lots.append(0)
        while len(asks_price_delta) < self.max_depth:
            asks_price_delta.append(0)
            asks_qty_lots.append(0)

        return OrderbookSnapshot(
            ts_ns=ts_ns,
            symbol=self.symbol,
            u=self.last_update_id,
            best_bid_ticks=best_bid_ticks,
            best_ask_ticks=best_ask_ticks,
            bids_price_delta=bids_price_delta,
            bids_qty_lots=bids_qty_lots,
            asks_price_delta=asks_price_delta,
            asks_qty_lots=asks_qty_lots,
            resync_epoch=self.resync_epoch,
        )

    @property
    def best_bid(self) -> Optional[tuple[int, int]]:
        """Return (price_ticks, qty_lots) of best bid or None."""
        if not self._bids:
            return None
        neg_price, qty = self._bids.peekitem(0)
        return (-neg_price, qty)

    @property
    def best_ask(self) -> Optional[tuple[int, int]]:
        """Return (price_ticks, qty_lots) of best ask or None."""
        if not self._asks:
            return None
        price, qty = self._asks.peekitem(0)
        return (price, qty)

    @property
    def bid_count(self) -> int:
        """Number of bid levels."""
        return len(self._bids)

    @property
    def ask_count(self) -> int:
        """Number of ask levels."""
        return len(self._asks)

    def __repr__(self) -> str:
        bid = self.best_bid
        ask = self.best_ask
        return (
            f"Orderbook({self.symbol}, "
            f"bids={self.bid_count}, asks={self.ask_count}, "
            f"best_bid={bid}, best_ask={ask}, "
            f"u={self.last_update_id})"
        )


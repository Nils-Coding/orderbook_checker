"""Tests for orderbook data structure."""

import pytest
from recorder.orderbook import Orderbook, OrderbookSnapshot


@pytest.fixture
def orderbook():
    """Create a test orderbook with simple price/qty conversion."""
    return Orderbook(
        symbol="BTCUSDT",
        max_depth=10,  # Small for testing
        price_to_ticks=lambda x: int(x * 10),  # 0.1 tick
        qty_to_lots=lambda x: int(x * 1000),   # 0.001 lot
    )


class TestOrderbookApply:
    """Tests for orderbook apply operations."""

    def test_apply_snapshot(self, orderbook):
        """Test applying REST snapshot."""
        snapshot = {
            "lastUpdateId": 12345,
            "bids": [
                ["100.0", "1.0"],
                ["99.9", "2.0"],
                ["99.8", "3.0"],
            ],
            "asks": [
                ["100.1", "1.5"],
                ["100.2", "2.5"],
                ["100.3", "3.5"],
            ],
        }

        orderbook.apply_snapshot(snapshot)

        assert orderbook.last_update_id == 12345
        assert orderbook.bid_count == 3
        assert orderbook.ask_count == 3

        # Best bid should be highest (100.0 = 1000 ticks)
        assert orderbook.best_bid == (1000, 1000)  # price_ticks, qty_lots

        # Best ask should be lowest (100.1 = 1001 ticks)
        assert orderbook.best_ask == (1001, 1500)

    def test_apply_diff_update(self, orderbook):
        """Test applying diff that updates existing levels."""
        # Setup initial state
        snapshot = {
            "lastUpdateId": 100,
            "bids": [["100.0", "1.0"]],
            "asks": [["100.1", "1.0"]],
        }
        orderbook.apply_snapshot(snapshot)

        # Apply diff that updates quantity
        diff = {
            "U": 101,
            "u": 101,
            "pu": 100,
            "b": [["100.0", "2.0"]],  # Update bid qty
            "a": [["100.1", "0.5"]],  # Update ask qty
        }

        orderbook.apply_diff(diff)

        assert orderbook.last_update_id == 101
        assert orderbook.best_bid == (1000, 2000)  # qty doubled
        assert orderbook.best_ask == (1001, 500)   # qty halved

    def test_apply_diff_delete(self, orderbook):
        """Test applying diff that deletes levels (qty=0)."""
        snapshot = {
            "lastUpdateId": 100,
            "bids": [["100.0", "1.0"], ["99.9", "2.0"]],
            "asks": [["100.1", "1.0"], ["100.2", "2.0"]],
        }
        orderbook.apply_snapshot(snapshot)

        # Delete top levels
        diff = {
            "U": 101,
            "u": 101,
            "pu": 100,
            "b": [["100.0", "0"]],  # Delete best bid
            "a": [["100.1", "0"]],  # Delete best ask
        }

        orderbook.apply_diff(diff)

        assert orderbook.bid_count == 1
        assert orderbook.ask_count == 1
        assert orderbook.best_bid == (999, 2000)   # Next level
        assert orderbook.best_ask == (1002, 2000)  # Next level

    def test_apply_diff_add_new(self, orderbook):
        """Test applying diff that adds new levels."""
        snapshot = {
            "lastUpdateId": 100,
            "bids": [["100.0", "1.0"]],
            "asks": [["100.2", "1.0"]],
        }
        orderbook.apply_snapshot(snapshot)

        # Add new levels
        diff = {
            "U": 101,
            "u": 101,
            "pu": 100,
            "b": [["100.1", "0.5"]],  # New higher bid at 100.1
            "a": [["100.1", "0.5"]],  # New lower ask at 100.1
        }

        orderbook.apply_diff(diff)

        # New best bid at 100.1 = 1001 ticks
        assert orderbook.best_bid[0] == 1001
        assert orderbook.bid_count == 2
        
        # New best ask at 100.1 = 1001 ticks
        assert orderbook.best_ask[0] == 1001
        assert orderbook.ask_count == 2

    def test_trim_on_apply(self):
        """Test that orderbook trims to max_depth after apply."""
        orderbook = Orderbook(
            symbol="TEST",
            max_depth=3,
            price_to_ticks=lambda x: int(x),
            qty_to_lots=lambda x: int(x),
        )

        # Apply snapshot with more than max_depth levels
        snapshot = {
            "lastUpdateId": 100,
            "bids": [
                ["100", "1"], ["99", "1"], ["98", "1"],
                ["97", "1"], ["96", "1"],  # These should be trimmed
            ],
            "asks": [
                ["101", "1"], ["102", "1"], ["103", "1"],
                ["104", "1"], ["105", "1"],  # These should be trimmed
            ],
        }
        orderbook.apply_snapshot(snapshot)

        assert orderbook.bid_count == 3
        assert orderbook.ask_count == 3

        # Best levels should be preserved
        assert orderbook.best_bid == (100, 1)
        assert orderbook.best_ask == (101, 1)


class TestOrderbookSnapshot:
    """Tests for snapshot generation."""

    def test_get_snapshot_basic(self, orderbook):
        """Test basic snapshot generation."""
        snapshot_data = {
            "lastUpdateId": 100,
            "bids": [["100.0", "1.0"], ["99.9", "2.0"]],
            "asks": [["100.1", "1.5"], ["100.2", "2.5"]],
        }
        orderbook.apply_snapshot(snapshot_data)

        snapshot = orderbook.get_snapshot()

        assert isinstance(snapshot, OrderbookSnapshot)
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.u == 100
        assert snapshot.best_bid_ticks == 1000
        assert snapshot.best_ask_ticks == 1001

    def test_snapshot_padding(self, orderbook):
        """Test that snapshot pads to max_depth levels."""
        snapshot_data = {
            "lastUpdateId": 100,
            "bids": [["100.0", "1.0"]],
            "asks": [["100.1", "1.0"]],
        }
        orderbook.apply_snapshot(snapshot_data)

        snapshot = orderbook.get_snapshot()

        # Should have max_depth (10) entries
        assert len(snapshot.bids_price_delta) == 10
        assert len(snapshot.bids_qty_lots) == 10
        assert len(snapshot.asks_price_delta) == 10
        assert len(snapshot.asks_qty_lots) == 10

        # First entry should be real data
        assert snapshot.bids_qty_lots[0] == 1000
        assert snapshot.asks_qty_lots[0] == 1000

        # Padding should be zeros
        assert snapshot.bids_qty_lots[1] == 0
        assert snapshot.asks_qty_lots[1] == 0
        assert snapshot.bids_price_delta[1] == 0
        assert snapshot.asks_price_delta[1] == 0

    def test_snapshot_deltas(self, orderbook):
        """Test that snapshot price deltas are relative to best price."""
        snapshot_data = {
            "lastUpdateId": 100,
            "bids": [
                ["100.0", "1.0"],  # best = 1000, delta = 0
                ["99.9", "2.0"],   # 999, delta = -1
                ["99.8", "3.0"],   # 998, delta = -2
            ],
            "asks": [
                ["100.1", "1.0"],  # best = 1001, delta = 0
                ["100.2", "2.0"],  # 1002, delta = 1
                ["100.3", "3.0"],  # 1003, delta = 2
            ],
        }
        orderbook.apply_snapshot(snapshot_data)

        snapshot = orderbook.get_snapshot()

        # Bid deltas should be 0, -1, -2, ...
        assert snapshot.bids_price_delta[0] == 0
        assert snapshot.bids_price_delta[1] == -1
        assert snapshot.bids_price_delta[2] == -2

        # Ask deltas should be 0, 1, 2, ...
        assert snapshot.asks_price_delta[0] == 0
        assert snapshot.asks_price_delta[1] == 1
        assert snapshot.asks_price_delta[2] == 2


class TestOrderbookClear:
    """Tests for orderbook clear operations."""

    def test_clear(self, orderbook):
        """Test clearing the orderbook."""
        snapshot = {
            "lastUpdateId": 100,
            "bids": [["100.0", "1.0"]],
            "asks": [["100.1", "1.0"]],
        }
        orderbook.apply_snapshot(snapshot)

        orderbook.clear()

        assert orderbook.bid_count == 0
        assert orderbook.ask_count == 0
        assert orderbook.last_update_id == 0
        assert orderbook.best_bid is None
        assert orderbook.best_ask is None


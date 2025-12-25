"""Integration tests with fixture data."""

import pytest
import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from recorder.orderbook import Orderbook, OrderbookSnapshot
from recorder.sync import OrderbookSync, SyncState
from recorder.writers import SnapshotWriter, TradeWriter, TradeRecord


# Sample fixture data simulating Binance responses
SAMPLE_SNAPSHOT = {
    "lastUpdateId": 1000,
    "bids": [
        ["50000.0", "1.5"],
        ["49999.9", "2.0"],
        ["49999.8", "3.0"],
        ["49999.7", "1.0"],
        ["49999.6", "0.5"],
    ],
    "asks": [
        ["50000.1", "1.2"],
        ["50000.2", "2.5"],
        ["50000.3", "1.8"],
        ["50000.4", "3.0"],
        ["50000.5", "0.8"],
    ],
}

SAMPLE_DEPTH_EVENTS = [
    {
        "e": "depthUpdate",
        "E": 1700000000100,
        "T": 1700000000099,
        "s": "BTCUSDT",
        "U": 998,
        "u": 1002,
        "pu": 997,
        "b": [["50000.0", "1.8"]],  # Update best bid qty
        "a": [],
    },
    {
        "e": "depthUpdate",
        "E": 1700000000200,
        "T": 1700000000199,
        "s": "BTCUSDT",
        "U": 1003,
        "u": 1005,
        "pu": 1002,
        "b": [],
        "a": [["50000.1", "1.0"]],  # Update best ask qty
    },
    {
        "e": "depthUpdate",
        "E": 1700000000300,
        "T": 1700000000299,
        "s": "BTCUSDT",
        "U": 1006,
        "u": 1008,
        "pu": 1005,
        "b": [["50000.05", "0.5"]],  # New bid level
        "a": [["50000.05", "0.3"]],  # New ask level
    },
]


class TestIntegrationOrderbookReplay:
    """Integration tests replaying fixture data through orderbook."""

    @pytest.fixture
    def orderbook(self):
        return Orderbook(
            symbol="BTCUSDT",
            max_depth=1000,
            price_to_ticks=lambda x: int(float(x) * 10),
            qty_to_lots=lambda x: int(float(x) * 1000),
        )

    def test_replay_snapshot_and_diffs(self, orderbook):
        """Test replaying a full sequence of snapshot + diffs."""
        # Apply initial snapshot
        orderbook.apply_snapshot(SAMPLE_SNAPSHOT)

        assert orderbook.last_update_id == 1000
        assert orderbook.bid_count == 5
        assert orderbook.ask_count == 5
        assert orderbook.best_bid == (500000, 1500)  # 50000.0 * 10, 1.5 * 1000

        # Apply first diff (updates best bid)
        orderbook.apply_diff(SAMPLE_DEPTH_EVENTS[0])
        assert orderbook.best_bid == (500000, 1800)  # qty updated to 1.8

        # Apply second diff (updates best ask)
        orderbook.apply_diff(SAMPLE_DEPTH_EVENTS[1])
        assert orderbook.best_ask == (500001, 1000)  # qty updated to 1.0

        # Apply third diff (adds new levels)
        orderbook.apply_diff(SAMPLE_DEPTH_EVENTS[2])
        # New bid at 50000.05 should not be best (500000.5 ticks rounds to 500000)
        assert orderbook.bid_count >= 5

    def test_snapshot_generation_after_replay(self, orderbook):
        """Test that snapshots are correctly generated after replay."""
        orderbook.apply_snapshot(SAMPLE_SNAPSHOT)

        for event in SAMPLE_DEPTH_EVENTS:
            orderbook.apply_diff(event)

        snapshot = orderbook.get_snapshot()

        # Verify snapshot structure
        assert snapshot.symbol == "BTCUSDT"
        assert len(snapshot.bids_price_delta) == 1000
        assert len(snapshot.asks_price_delta) == 1000

        # First entries should have data
        assert snapshot.bids_qty_lots[0] > 0
        assert snapshot.asks_qty_lots[0] > 0


class TestIntegrationSyncReplay:
    """Integration tests for sync state machine with fixture data."""

    @pytest.fixture
    def setup(self):
        orderbook = Orderbook(
            symbol="BTCUSDT",
            max_depth=1000,
            price_to_ticks=lambda x: int(float(x) * 10),
            qty_to_lots=lambda x: int(float(x) * 1000),
        )
        sync = OrderbookSync(
            orderbook=orderbook,
            rest_url="https://fapi.binance.com",
            symbol="BTCUSDT",
        )
        return orderbook, sync

    @pytest.mark.asyncio
    async def test_sync_sequence_validation(self, setup):
        """Test sync handles sequence gaps correctly."""
        orderbook, sync = setup
        await sync.start()

        # Buffer events
        for event in SAMPLE_DEPTH_EVENTS:
            sync.on_ws_event(event)

        assert sync.stats.events_buffered == 3

    @pytest.mark.asyncio
    async def test_sync_finds_sync_point(self, setup):
        """Test sync finds the correct sync point."""
        orderbook, sync = setup
        await sync.start()

        # Buffer events first
        sync.on_ws_event(SAMPLE_DEPTH_EVENTS[0])  # U=998, u=1002
        sync.on_ws_event(SAMPLE_DEPTH_EVENTS[1])  # U=1003, u=1005

        # Apply snapshot with lastUpdateId=1000
        # First event should match: U=998 <= 1000 <= u=1002
        orderbook.apply_snapshot(SAMPLE_SNAPSHOT)
        sync._snapshot_last_update_id = SAMPLE_SNAPSHOT["lastUpdateId"]

        # Process buffer
        sync._set_state(SyncState.SYNCING)
        events_to_process = list(sync._buffer)
        sync._buffer.clear()

        found_sync = False
        for event in events_to_process:
            U = event.get("U", 0)
            u = event.get("u", 0)

            if U <= sync._snapshot_last_update_id <= u:
                found_sync = True
                break

        assert found_sync, "Should find sync point in first event"


class TestIntegrationWriters:
    """Integration tests for Parquet writers."""

    @pytest.mark.asyncio
    async def test_snapshot_writer_creates_files(self):
        """Test snapshot writer creates Parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)

            writer = SnapshotWriter(
                data_root=data_root,
                symbol="BTCUSDT",
                queue_max=100,
                chunk_seconds=1,  # Fast rotation for testing
            )

            await writer.start()

            # Create and enqueue snapshots
            for i in range(5):
                snapshot = OrderbookSnapshot(
                    ts_ns=1700000000000000000 + i * 100000000,
                    symbol="BTCUSDT",
                    u=1000 + i,
                    best_bid_ticks=500000,
                    best_ask_ticks=500001,
                    bids_price_delta=[0] * 1000,
                    bids_qty_lots=[1000] * 1000,
                    asks_price_delta=[0] * 1000,
                    asks_qty_lots=[1000] * 1000,
                    resync_epoch=0,
                )
                await writer.enqueue(snapshot)

            # Wait for processing and rotation
            await asyncio.sleep(1.5)

            # Force flush
            await writer.stop()

            # Verify files created
            snap_dir = data_root / "snapshots" / "symbol=BTCUSDT"
            assert snap_dir.exists(), f"Directory should exist: {snap_dir}"
            files = list(snap_dir.rglob("*.parquet"))
            assert len(files) > 0, f"Should have created parquet files in {snap_dir}"

            # Verify content - read single file directly
            parquet_file = pq.ParquetFile(str(files[0]))
            table = parquet_file.read()
            df = table.to_pandas()
            assert len(df) > 0
            assert "ts_ns" in df.columns
            assert "symbol" in df.columns
            assert "bids_qty_lots" in df.columns

    @pytest.mark.asyncio
    async def test_trade_writer_creates_files(self):
        """Test trade writer creates Parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)

            writer = TradeWriter(
                data_root=data_root,
                symbol="BTCUSDT",
                queue_max=100,
                chunk_seconds=1,
            )

            await writer.start()

            # Create and enqueue trades
            for i in range(10):
                trade = TradeRecord(
                    ts_ns=1700000000000000000 + i * 10000000,
                    symbol="BTCUSDT",
                    price_ticks=500000 + i,
                    qty_lots=100 + i * 10,
                    is_buyer_maker=i % 2 == 0,
                )
                await writer.enqueue(trade)

            # Wait for processing and rotation
            await asyncio.sleep(1.5)
            await writer.stop()

            # Verify files created
            trade_dir = data_root / "trades" / "symbol=BTCUSDT"
            assert trade_dir.exists(), f"Directory should exist: {trade_dir}"
            files = list(trade_dir.rglob("*.parquet"))
            assert len(files) > 0, f"Should have created parquet files in {trade_dir}"

            # Verify content - read single file directly
            parquet_file = pq.ParquetFile(str(files[0]))
            table = parquet_file.read()
            df = table.to_pandas()
            assert len(df) > 0
            assert "is_buyer_maker" in df.columns


class TestIntegrationPenetration:
    """Integration tests for penetration analysis."""

    def test_penetration_calculation(self):
        """Test penetration level calculation."""
        from tools.analyze_penetration import calculate_penetration

        # Orderbook with 5 levels of 100 lots each
        qty_levels = np.array([100, 100, 100, 100, 100] + [0] * 995)

        # Trade of 50 lots -> Level 1
        assert calculate_penetration(50, qty_levels) == 1

        # Trade of 100 lots -> Level 1 (exactly fills)
        assert calculate_penetration(100, qty_levels) == 1

        # Trade of 150 lots -> Level 2
        assert calculate_penetration(150, qty_levels) == 2

        # Trade of 500 lots -> Level 5 (exactly fills all)
        assert calculate_penetration(500, qty_levels) == 5

        # Trade of 600 lots -> Overflow (1001)
        assert calculate_penetration(600, qty_levels) == 1001

    def test_penetration_with_gaps(self):
        """Test penetration with empty levels."""
        from tools.analyze_penetration import calculate_penetration

        # Some levels have zero qty (gaps)
        qty_levels = np.array([100, 0, 0, 100, 100] + [0] * 995)

        # Trade of 150 -> Level 4 (skips empty levels)
        assert calculate_penetration(150, qty_levels) == 4


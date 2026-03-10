"""Parquet writers for snapshots and trades with bounded queues."""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .orderbook import OrderbookSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Trade data record (supports both @trade and @aggTrade)."""

    ts_ns: int
    symbol: str
    price_ticks: int
    qty_lots: int
    is_buyer_maker: bool
    # aggTrade specific fields (0 if from @trade stream)
    agg_trade_id: int = 0
    first_trade_id: int = 0
    last_trade_id: int = 0


class SnapshotWriter:
    """
    Async writer for orderbook snapshots to Parquet files.
    
    - Uses bounded queue for backpressure
    - Writes chunked files (rotation by time)
    - ZSTD compression
    - Non-blocking enqueue
    """

    def __init__(
        self,
        data_root: Path,
        symbol: str,
        queue_max: int = 600,
        chunk_seconds: int = 60,
        fail_on_backpressure: bool = True,
    ):
        self.data_root = data_root
        self.symbol = symbol
        self.queue_max = queue_max
        self.chunk_seconds = chunk_seconds
        self.fail_on_backpressure = fail_on_backpressure

        self._queue: asyncio.Queue[Optional[OrderbookSnapshot]] = asyncio.Queue(maxsize=queue_max)
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Current chunk state
        self._chunk_start: Optional[datetime] = None
        self._chunk_rows: list[dict] = []
        self._current_path: Optional[Path] = None

        # Stats
        self.snapshots_written = 0
        self.chunks_written = 0
        self.queue_full_count = 0

    def _get_output_path(self, ts: datetime) -> Path:
        """Get output path for given timestamp."""
        return (
            self.data_root
            / "snapshots"
            / f"symbol={self.symbol}"
            / f"date={ts.strftime('%Y-%m-%d')}"
            / f"hour={ts.strftime('%H')}"
        )

    async def start(self) -> None:
        """Start the writer task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"SnapshotWriter started for {self.symbol}")

    async def stop(self) -> None:
        """Stop the writer task and flush remaining data."""
        self._running = False
        # Send sentinel to signal shutdown
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=30)
            except asyncio.TimeoutError:
                logger.warning("Writer shutdown timed out")
                self._task.cancel()
            except asyncio.CancelledError:
                pass

        # Final flush
        if self._chunk_rows:
            await self._flush_chunk()

        logger.info(f"SnapshotWriter stopped: {self.snapshots_written} snapshots, {self.chunks_written} chunks")

    async def enqueue(self, snapshot: OrderbookSnapshot) -> bool:
        """
        Enqueue snapshot for writing.
        
        Returns True if enqueued, False if queue full.
        Non-blocking.
        """
        try:
            self._queue.put_nowait(snapshot)
            return True
        except asyncio.QueueFull:
            self.queue_full_count += 1
            if self.fail_on_backpressure:
                logger.error("Snapshot queue full - backpressure triggered")
                raise RuntimeError("Snapshot writer queue full - aborting to prevent data gaps")
            else:
                logger.warning("Snapshot queue full - dropping snapshot")
                return False

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def queue_fill_pct(self) -> float:
        return (self._queue.qsize() / self.queue_max) * 100

    async def _run(self) -> None:
        """Main writer loop."""
        while self._running or not self._queue.empty():
            try:
                snapshot = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )

                if snapshot is None:
                    # Sentinel received
                    break

                await self._write_snapshot(snapshot)

            except asyncio.TimeoutError:
                # Check if chunk needs rotation
                if self._chunk_rows and self._should_rotate():
                    await self._flush_chunk()

    async def _write_snapshot(self, snapshot: OrderbookSnapshot) -> None:
        """Process a single snapshot."""
        ts = datetime.fromtimestamp(snapshot.ts_ns / 1e9)

        # Check if we need to rotate chunk
        if self._chunk_start is None:
            self._chunk_start = ts
            self._current_path = self._get_output_path(ts)
        elif self._should_rotate() or self._get_output_path(ts) != self._current_path:
            await self._flush_chunk()
            self._chunk_start = ts
            self._current_path = self._get_output_path(ts)

        # Add row
        self._chunk_rows.append({
            "ts_ns": snapshot.ts_ns,
            "symbol": snapshot.symbol,
            "u": snapshot.u,
            "best_bid_ticks": snapshot.best_bid_ticks,
            "best_ask_ticks": snapshot.best_ask_ticks,
            "bids_price_delta": snapshot.bids_price_delta,
            "bids_qty_lots": snapshot.bids_qty_lots,
            "asks_price_delta": snapshot.asks_price_delta,
            "asks_qty_lots": snapshot.asks_qty_lots,
            "resync_epoch": snapshot.resync_epoch,
        })

        self.snapshots_written += 1

    def _should_rotate(self) -> bool:
        """Check if chunk should be rotated."""
        if self._chunk_start is None:
            return False
        elapsed = (datetime.now() - self._chunk_start).total_seconds()
        return elapsed >= self.chunk_seconds

    async def _flush_chunk(self) -> None:
        """Flush current chunk to Parquet file."""
        if not self._chunk_rows:
            return

        # Create directory
        output_dir = self._current_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find next part number
        existing = list(output_dir.glob("part-*.parquet"))
        part_num = len(existing)
        output_file = output_dir / f"part-{part_num:04d}.parquet"

        # Build table
        table = self._build_table()

        # Write with ZSTD compression
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pq.write_table(
                table,
                output_file,
                compression="zstd",
                compression_level=3,
            )
        )

        logger.info(f"Wrote {len(self._chunk_rows)} snapshots to {output_file}")
        self.chunks_written += 1
        self._chunk_rows = []

    def _build_table(self) -> pa.Table:
        """Build PyArrow table from chunk rows."""
        # Schema with list types for price/qty arrays
        schema = pa.schema([
            ("ts_ns", pa.int64()),
            ("symbol", pa.string()),
            ("u", pa.int64()),
            ("best_bid_ticks", pa.int64()),
            ("best_ask_ticks", pa.int64()),
            ("bids_price_delta", pa.list_(pa.int32())),
            ("bids_qty_lots", pa.list_(pa.int64())),
            ("asks_price_delta", pa.list_(pa.int32())),
            ("asks_qty_lots", pa.list_(pa.int64())),
            ("resync_epoch", pa.int32()),
        ])

        arrays = {
            "ts_ns": pa.array([r["ts_ns"] for r in self._chunk_rows], type=pa.int64()),
            "symbol": pa.array([r["symbol"] for r in self._chunk_rows], type=pa.string()),
            "u": pa.array([r["u"] for r in self._chunk_rows], type=pa.int64()),
            "best_bid_ticks": pa.array([r["best_bid_ticks"] for r in self._chunk_rows], type=pa.int64()),
            "best_ask_ticks": pa.array([r["best_ask_ticks"] for r in self._chunk_rows], type=pa.int64()),
            "bids_price_delta": pa.array(
                [np.array(r["bids_price_delta"], dtype=np.int32) for r in self._chunk_rows],
                type=pa.list_(pa.int32())
            ),
            "bids_qty_lots": pa.array(
                [np.array(r["bids_qty_lots"], dtype=np.int64) for r in self._chunk_rows],
                type=pa.list_(pa.int64())
            ),
            "asks_price_delta": pa.array(
                [np.array(r["asks_price_delta"], dtype=np.int32) for r in self._chunk_rows],
                type=pa.list_(pa.int32())
            ),
            "asks_qty_lots": pa.array(
                [np.array(r["asks_qty_lots"], dtype=np.int64) for r in self._chunk_rows],
                type=pa.list_(pa.int64())
            ),
            "resync_epoch": pa.array([r["resync_epoch"] for r in self._chunk_rows], type=pa.int32()),
        }

        return pa.table(arrays, schema=schema)


class TradeWriter:
    """
    Async writer for trades to Parquet files.
    
    - Uses bounded queue for backpressure
    - Writes chunked files (rotation by time)
    - ZSTD compression
    """

    def __init__(
        self,
        data_root: Path,
        symbol: str,
        queue_max: int = 600,
        chunk_seconds: int = 60,
        fail_on_backpressure: bool = True,
    ):
        self.data_root = data_root
        self.symbol = symbol
        self.queue_max = queue_max
        self.chunk_seconds = chunk_seconds
        self.fail_on_backpressure = fail_on_backpressure

        self._queue: asyncio.Queue[Optional[TradeRecord]] = asyncio.Queue(maxsize=queue_max)
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Current chunk state
        self._chunk_start: Optional[datetime] = None
        self._chunk_rows: list[dict] = []
        self._current_path: Optional[Path] = None

        # Stats
        self.trades_written = 0
        self.chunks_written = 0
        self.queue_full_count = 0

    def _get_output_path(self, ts: datetime) -> Path:
        """Get output path for given timestamp."""
        return (
            self.data_root
            / "trades"
            / f"symbol={self.symbol}"
            / f"date={ts.strftime('%Y-%m-%d')}"
            / f"hour={ts.strftime('%H')}"
        )

    async def start(self) -> None:
        """Start the writer task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"TradeWriter started for {self.symbol}")

    async def stop(self) -> None:
        """Stop the writer task and flush remaining data."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=30)
            except asyncio.TimeoutError:
                logger.warning("Trade writer shutdown timed out")
                self._task.cancel()
            except asyncio.CancelledError:
                pass

        if self._chunk_rows:
            await self._flush_chunk()

        logger.info(f"TradeWriter stopped: {self.trades_written} trades, {self.chunks_written} chunks")

    async def enqueue(self, trade: TradeRecord) -> bool:
        """
        Enqueue trade for writing.
        
        Returns True if enqueued, False if queue full.
        """
        try:
            self._queue.put_nowait(trade)
            return True
        except asyncio.QueueFull:
            self.queue_full_count += 1
            if self.fail_on_backpressure:
                logger.error("Trade queue full - backpressure triggered")
                raise RuntimeError("Trade writer queue full - aborting to prevent data gaps")
            else:
                logger.warning("Trade queue full - dropping trade")
                return False

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def queue_fill_pct(self) -> float:
        return (self._queue.qsize() / self.queue_max) * 100

    async def _run(self) -> None:
        """Main writer loop."""
        while self._running or not self._queue.empty():
            try:
                trade = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )

                if trade is None:
                    break

                await self._write_trade(trade)

            except asyncio.TimeoutError:
                if self._chunk_rows and self._should_rotate():
                    await self._flush_chunk()

    async def _write_trade(self, trade: TradeRecord) -> None:
        """Process a single trade."""
        ts = datetime.fromtimestamp(trade.ts_ns / 1e9)

        if self._chunk_start is None:
            self._chunk_start = ts
            self._current_path = self._get_output_path(ts)
        elif self._should_rotate() or self._get_output_path(ts) != self._current_path:
            await self._flush_chunk()
            self._chunk_start = ts
            self._current_path = self._get_output_path(ts)

        self._chunk_rows.append({
            "ts_ns": trade.ts_ns,
            "symbol": trade.symbol,
            "price_ticks": trade.price_ticks,
            "qty_lots": trade.qty_lots,
            "is_buyer_maker": trade.is_buyer_maker,
            "agg_trade_id": trade.agg_trade_id,
            "first_trade_id": trade.first_trade_id,
            "last_trade_id": trade.last_trade_id,
        })

        self.trades_written += 1

    def _should_rotate(self) -> bool:
        """Check if chunk should be rotated."""
        if self._chunk_start is None:
            return False
        elapsed = (datetime.now() - self._chunk_start).total_seconds()
        return elapsed >= self.chunk_seconds

    async def _flush_chunk(self) -> None:
        """Flush current chunk to Parquet file."""
        if not self._chunk_rows:
            return

        output_dir = self._current_path
        output_dir.mkdir(parents=True, exist_ok=True)

        existing = list(output_dir.glob("part-*.parquet"))
        part_num = len(existing)
        output_file = output_dir / f"part-{part_num:04d}.parquet"

        table = self._build_table()

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pq.write_table(
                table,
                output_file,
                compression="zstd",
                compression_level=3,
            )
        )

        logger.info(f"Wrote {len(self._chunk_rows)} trades to {output_file}")
        self.chunks_written += 1
        self._chunk_rows = []

    def _build_table(self) -> pa.Table:
        """Build PyArrow table from chunk rows."""
        schema = pa.schema([
            ("ts_ns", pa.int64()),
            ("symbol", pa.string()),
            ("price_ticks", pa.int64()),
            ("qty_lots", pa.int64()),
            ("is_buyer_maker", pa.bool_()),
            ("agg_trade_id", pa.int64()),
            ("first_trade_id", pa.int64()),
            ("last_trade_id", pa.int64()),
        ])

        arrays = {
            "ts_ns": pa.array([r["ts_ns"] for r in self._chunk_rows], type=pa.int64()),
            "symbol": pa.array([r["symbol"] for r in self._chunk_rows], type=pa.string()),
            "price_ticks": pa.array([r["price_ticks"] for r in self._chunk_rows], type=pa.int64()),
            "qty_lots": pa.array([r["qty_lots"] for r in self._chunk_rows], type=pa.int64()),
            "is_buyer_maker": pa.array([r["is_buyer_maker"] for r in self._chunk_rows], type=pa.bool_()),
            "agg_trade_id": pa.array([r["agg_trade_id"] for r in self._chunk_rows], type=pa.int64()),
            "first_trade_id": pa.array([r["first_trade_id"] for r in self._chunk_rows], type=pa.int64()),
            "last_trade_id": pa.array([r["last_trade_id"] for r in self._chunk_rows], type=pa.int64()),
        }

        return pa.table(arrays, schema=schema)


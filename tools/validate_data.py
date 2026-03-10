#!/usr/bin/env python3
"""
Data validation tools for orderbook recorder.

Includes:
- Gap detection in snapshot timestamps
- Trade ID sequence validation
- Spread sanity checks
- Cross-validation with Binance REST API
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class GapReport:
    """Report of gaps found in snapshot data."""
    
    total_snapshots: int
    expected_interval_ms: int
    tolerance_ms: int
    gaps_found: int
    max_gap_ms: float
    total_gap_time_s: float
    gap_locations: list[tuple[datetime, datetime, float]]  # (start, end, duration_ms)
    
    def print_report(self) -> None:
        """Print formatted gap report."""
        print("\n" + "=" * 60)
        print("GAP DETECTION REPORT")
        print("=" * 60)
        print(f"Total snapshots analyzed: {self.total_snapshots:,}")
        print(f"Expected interval: {self.expected_interval_ms}ms (tolerance: {self.tolerance_ms}ms)")
        print(f"\nGaps found: {self.gaps_found}")
        if self.gaps_found > 0:
            print(f"Max gap: {self.max_gap_ms:.1f}ms ({self.max_gap_ms/1000:.2f}s)")
            print(f"Total gap time: {self.total_gap_time_s:.2f}s")
            completeness = 100 * (1 - self.total_gap_time_s / (self.total_snapshots * self.expected_interval_ms / 1000))
            print(f"Data completeness: {completeness:.2f}%")
            
            if self.gap_locations:
                print(f"\nTop 10 largest gaps:")
                sorted_gaps = sorted(self.gap_locations, key=lambda x: x[2], reverse=True)[:10]
                for i, (start, end, duration) in enumerate(sorted_gaps, 1):
                    print(f"  {i}. {start.strftime('%Y-%m-%d %H:%M:%S')} - {duration:.0f}ms")
        else:
            print("No gaps detected - data is complete!")
        print("=" * 60)


@dataclass
class TradeIdReport:
    """Report of trade ID sequence validation."""
    
    total_trades: int
    agg_trade_id_min: int
    agg_trade_id_max: int
    gaps_found: int
    missing_ids: list[tuple[int, int]]  # (start, end) of missing ranges
    
    def print_report(self) -> None:
        """Print formatted trade ID report."""
        print("\n" + "=" * 60)
        print("TRADE ID SEQUENCE REPORT")
        print("=" * 60)
        print(f"Total trades analyzed: {self.total_trades:,}")
        print(f"Trade ID range: {self.agg_trade_id_min:,} - {self.agg_trade_id_max:,}")
        expected = self.agg_trade_id_max - self.agg_trade_id_min + 1
        print(f"Expected count: {expected:,}")
        print(f"\nSequence gaps: {self.gaps_found}")
        if self.gaps_found > 0:
            print(f"Missing trade IDs: {sum(e - s for s, e in self.missing_ids):,}")
            completeness = 100 * self.total_trades / expected
            print(f"Completeness: {completeness:.2f}%")
            
            if self.missing_ids[:10]:
                print(f"\nFirst 10 gaps:")
                for i, (start, end) in enumerate(self.missing_ids[:10], 1):
                    print(f"  {i}. IDs {start:,} to {end:,} ({end - start:,} missing)")
        else:
            print("No gaps - trade sequence is complete!")
        print("=" * 60)


@dataclass
class SpreadReport:
    """Report of spread sanity checks."""
    
    total_snapshots: int
    crossed_book_count: int
    zero_spread_count: int
    min_spread_ticks: int
    max_spread_ticks: int
    avg_spread_ticks: float
    crossed_book_times: list[datetime]
    
    def print_report(self) -> None:
        """Print formatted spread report."""
        print("\n" + "=" * 60)
        print("SPREAD SANITY REPORT")
        print("=" * 60)
        print(f"Total snapshots analyzed: {self.total_snapshots:,}")
        print(f"\nCrossed book events: {self.crossed_book_count}")
        print(f"Zero spread events: {self.zero_spread_count}")
        print(f"\nSpread statistics (in ticks):")
        print(f"  Min: {self.min_spread_ticks}")
        print(f"  Max: {self.max_spread_ticks}")
        print(f"  Avg: {self.avg_spread_ticks:.2f}")
        
        if self.crossed_book_count > 0:
            print(f"\nWARNING: Crossed book detected at:")
            for ts in self.crossed_book_times[:10]:
                print(f"  - {ts}")
        else:
            print("\nAll spreads valid - no crossed books!")
        print("=" * 60)


def load_snapshots(data_root: Path, symbol: str, date: str) -> pd.DataFrame:
    """Load all snapshots for a symbol/date."""
    base_path = data_root / "snapshots" / f"symbol={symbol}" / f"date={date}"
    
    if not base_path.exists():
        raise FileNotFoundError(f"No data found at {base_path}")
    
    parquet_files = list(base_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {base_path}")
    
    logger.info(f"Loading {len(parquet_files)} snapshot files from {base_path}")
    
    dfs = []
    for f in sorted(parquet_files):
        pf = pq.ParquetFile(f)
        df = pf.read().to_pandas()
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("ts_ns").reset_index(drop=True)
    
    logger.info(f"Loaded {len(combined):,} snapshots")
    return combined


def load_trades(data_root: Path, symbol: str, date: str) -> pd.DataFrame:
    """Load all trades for a symbol/date."""
    base_path = data_root / "trades" / f"symbol={symbol}" / f"date={date}"
    
    if not base_path.exists():
        raise FileNotFoundError(f"No data found at {base_path}")
    
    parquet_files = list(base_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {base_path}")
    
    logger.info(f"Loading {len(parquet_files)} trade files from {base_path}")
    
    dfs = []
    for f in sorted(parquet_files):
        pf = pq.ParquetFile(f)
        df = pf.read().to_pandas()
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("ts_ns").reset_index(drop=True)
    
    logger.info(f"Loaded {len(combined):,} trades")
    return combined


def detect_gaps(
    snapshots: pd.DataFrame,
    expected_interval_ms: int = 100,
    tolerance_ms: int = 50,
) -> GapReport:
    """Detect gaps in snapshot timestamps."""
    
    timestamps = snapshots["ts_ns"].values
    deltas_ms = np.diff(timestamps) / 1_000_000
    
    threshold = expected_interval_ms + tolerance_ms
    gap_mask = deltas_ms > threshold
    
    gap_indices = np.where(gap_mask)[0]
    gap_durations = deltas_ms[gap_mask]
    
    gap_locations = []
    for idx in gap_indices:
        start_ts = pd.Timestamp(timestamps[idx], unit="ns")
        end_ts = pd.Timestamp(timestamps[idx + 1], unit="ns")
        duration = deltas_ms[idx]
        gap_locations.append((start_ts.to_pydatetime(), end_ts.to_pydatetime(), duration))
    
    return GapReport(
        total_snapshots=len(snapshots),
        expected_interval_ms=expected_interval_ms,
        tolerance_ms=tolerance_ms,
        gaps_found=len(gap_durations),
        max_gap_ms=float(gap_durations.max()) if len(gap_durations) > 0 else 0.0,
        total_gap_time_s=float(gap_durations.sum()) / 1000 if len(gap_durations) > 0 else 0.0,
        gap_locations=gap_locations,
    )


def validate_trade_ids(trades: pd.DataFrame) -> Optional[TradeIdReport]:
    """Validate trade ID sequence for gaps."""
    
    if "agg_trade_id" not in trades.columns:
        logger.warning("agg_trade_id column not found - skipping trade ID validation")
        return None
    
    if trades["agg_trade_id"].isna().all() or (trades["agg_trade_id"] == 0).all():
        logger.warning("agg_trade_id values are all 0/null - data from @trade stream, not @aggTrade")
        return None
    
    ids = trades["agg_trade_id"].sort_values().values
    id_min = int(ids[0])
    id_max = int(ids[-1])
    
    diffs = np.diff(ids)
    gap_mask = diffs > 1
    gap_indices = np.where(gap_mask)[0]
    
    missing_ranges = []
    for idx in gap_indices:
        start = int(ids[idx]) + 1
        end = int(ids[idx + 1])
        missing_ranges.append((start, end))
    
    return TradeIdReport(
        total_trades=len(trades),
        agg_trade_id_min=id_min,
        agg_trade_id_max=id_max,
        gaps_found=len(missing_ranges),
        missing_ids=missing_ranges,
    )


def check_spreads(snapshots: pd.DataFrame) -> SpreadReport:
    """Check for spread anomalies."""
    
    spreads = snapshots["best_ask_ticks"] - snapshots["best_bid_ticks"]
    
    crossed_mask = spreads < 0
    crossed_count = crossed_mask.sum()
    crossed_times = snapshots.loc[crossed_mask, "ts_ns"].apply(
        lambda x: pd.Timestamp(x, unit="ns").to_pydatetime()
    ).tolist()
    
    zero_spread_count = (spreads == 0).sum()
    
    valid_spreads = spreads[spreads > 0]
    
    return SpreadReport(
        total_snapshots=len(snapshots),
        crossed_book_count=int(crossed_count),
        zero_spread_count=int(zero_spread_count),
        min_spread_ticks=int(valid_spreads.min()) if len(valid_spreads) > 0 else 0,
        max_spread_ticks=int(valid_spreads.max()) if len(valid_spreads) > 0 else 0,
        avg_spread_ticks=float(valid_spreads.mean()) if len(valid_spreads) > 0 else 0.0,
        crossed_book_times=crossed_times,
    )


async def fetch_binance_aggtrades(
    symbol: str,
    start_time: int,
    end_time: int,
) -> list[dict]:
    """Fetch aggregated trades from Binance REST API."""
    
    url = "https://fapi.binance.com/fapi/v1/aggTrades"
    all_trades = []
    
    async with aiohttp.ClientSession() as session:
        current_start = start_time
        
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": min(current_start + 3600000, end_time),  # 1 hour chunks
                "limit": 1000,
            }
            
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Binance API error: {resp.status}")
                    break
                    
                data = await resp.json()
                if not data:
                    break
                    
                all_trades.extend(data)
                
                last_time = data[-1]["T"]
                if last_time >= end_time or len(data) < 1000:
                    break
                    
                current_start = last_time + 1
    
    return all_trades


async def compare_with_binance(
    trades: pd.DataFrame,
    symbol: str,
) -> None:
    """Compare local trades with Binance REST API."""
    
    if "agg_trade_id" not in trades.columns or (trades["agg_trade_id"] == 0).all():
        logger.warning("Cannot compare with Binance - no agg_trade_id in local data")
        return
    
    start_ts = trades["ts_ns"].min() // 1_000_000
    end_ts = trades["ts_ns"].max() // 1_000_000
    
    logger.info(f"Fetching Binance trades from {start_ts} to {end_ts}...")
    binance_trades = await fetch_binance_aggtrades(symbol, start_ts, end_ts)
    
    print("\n" + "=" * 60)
    print("BINANCE COMPARISON REPORT")
    print("=" * 60)
    print(f"Local trades: {len(trades):,}")
    print(f"Binance trades: {len(binance_trades):,}")
    
    if binance_trades:
        binance_ids = set(t["a"] for t in binance_trades)
        local_ids = set(trades["agg_trade_id"].values)
        
        missing = binance_ids - local_ids
        extra = local_ids - binance_ids
        
        print(f"\nMissing from local: {len(missing)}")
        print(f"Extra in local: {len(extra)}")
        
        if missing:
            completeness = 100 * (1 - len(missing) / len(binance_ids))
            print(f"Completeness: {completeness:.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate orderbook recorder data")
    parser.add_argument("--data-root", type=Path, default=Path("./data"), help="Data root directory")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--date", type=str, help="Date to analyze (YYYY-MM-DD), defaults to yesterday")
    parser.add_argument("--interval-ms", type=int, default=100, help="Expected snapshot interval in ms")
    parser.add_argument("--tolerance-ms", type=int, default=50, help="Gap detection tolerance in ms")
    parser.add_argument("--compare-binance", action="store_true", help="Compare with Binance REST API")
    
    args = parser.parse_args()
    
    if args.date is None:
        args.date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(f"Validating data for {args.symbol} on {args.date}")
    
    try:
        snapshots = load_snapshots(args.data_root, args.symbol, args.date)
        gap_report = detect_gaps(snapshots, args.interval_ms, args.tolerance_ms)
        gap_report.print_report()
        
        spread_report = check_spreads(snapshots)
        spread_report.print_report()
    except FileNotFoundError as e:
        logger.warning(f"Snapshot data not found: {e}")
    
    try:
        trades = load_trades(args.data_root, args.symbol, args.date)
        trade_report = validate_trade_ids(trades)
        if trade_report:
            trade_report.print_report()
        
        if args.compare_binance:
            asyncio.run(compare_with_binance(trades, args.symbol))
    except FileNotFoundError as e:
        logger.warning(f"Trade data not found: {e}")


if __name__ == "__main__":
    main()

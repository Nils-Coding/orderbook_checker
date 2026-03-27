#!/usr/bin/env python3
"""
Calibrate timestamp offset between trades and orderbook snapshots.

Analyzes the dynamic offset over time using a sliding window approach.
For each time window, finds the offset that minimizes "outside spread" trades.

Output:
- Offset curve over time (CSV)
- Statistics about offset stability
- Visualization data for plotting
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of timestamp calibration for a time window."""
    
    window_start_ns: int
    window_end_ns: int
    optimal_offset_ms: float
    outside_spread_rate: float
    trade_count: int
    snapshot_count: int


def load_data_lazy(data_root: Path, symbol: str, date: str) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Load snapshots and trades as lazy frames for memory efficiency."""
    
    snap_path = data_root / "snapshots" / symbol / date
    trade_path = data_root / "trades" / symbol / date
    
    if not snap_path.exists():
        raise FileNotFoundError(f"Snapshot path not found: {snap_path}")
    if not trade_path.exists():
        raise FileNotFoundError(f"Trade path not found: {trade_path}")
    
    logger.info(f"Scanning snapshots from {snap_path}")
    snapshots = pl.scan_parquet(f"{snap_path}/**/*.parquet")
    
    logger.info(f"Scanning trades from {trade_path}")
    trades = pl.scan_parquet(f"{trade_path}/**/*.parquet")
    
    return snapshots, trades


def calculate_outside_rate_for_offset(
    trades_ts: np.ndarray,
    trades_price: np.ndarray,
    snap_ts: np.ndarray,
    snap_bid: np.ndarray,
    snap_ask: np.ndarray,
    offset_ns: int,
) -> float:
    """
    Calculate the rate of trades outside spread for a given timestamp offset.
    
    Args:
        offset_ns: Offset to ADD to snapshot timestamps (positive = shift forward)
    """
    # Shift snapshot timestamps
    shifted_snap_ts = snap_ts + offset_ns
    
    outside_count = 0
    total_count = len(trades_ts)
    
    for i in range(total_count):
        trade_ts = trades_ts[i]
        trade_price = trades_price[i]
        
        # Find nearest snapshot BEFORE the trade
        idx = np.searchsorted(shifted_snap_ts, trade_ts, side="right") - 1
        
        if idx < 0 or idx >= len(snap_bid):
            continue
        
        bid = snap_bid[idx]
        ask = snap_ask[idx]
        
        # Check if outside spread
        if trade_price < bid or trade_price > ask:
            outside_count += 1
    
    return outside_count / total_count if total_count > 0 else 1.0


def find_optimal_offset(
    trades_ts: np.ndarray,
    trades_price: np.ndarray,
    snap_ts: np.ndarray,
    snap_bid: np.ndarray,
    snap_ask: np.ndarray,
    search_range_ms: int = 500,
    step_ms: int = 10,
) -> tuple[float, float]:
    """
    Find the optimal timestamp offset that minimizes outside-spread rate.
    
    Returns:
        (optimal_offset_ms, best_outside_rate)
    """
    best_offset_ms = 0
    best_rate = 1.0
    
    for offset_ms in range(-search_range_ms, search_range_ms + 1, step_ms):
        offset_ns = offset_ms * 1_000_000
        rate = calculate_outside_rate_for_offset(
            trades_ts, trades_price, snap_ts, snap_bid, snap_ask, offset_ns
        )
        
        if rate < best_rate:
            best_rate = rate
            best_offset_ms = offset_ms
    
    return float(best_offset_ms), best_rate


def calibrate_window(
    trades_df: pl.DataFrame,
    snapshots_df: pl.DataFrame,
    window_start_ns: int,
    window_end_ns: int,
    search_range_ms: int = 500,
    step_ms: int = 10,
) -> Optional[CalibrationResult]:
    """Calibrate offset for a single time window."""
    
    # Filter to window
    window_trades = trades_df.filter(
        (pl.col("ts_ns") >= window_start_ns) & (pl.col("ts_ns") < window_end_ns)
    )
    
    # Get snapshots with some margin before and after
    margin_ns = 1_000_000_000  # 1 second margin
    window_snaps = snapshots_df.filter(
        (pl.col("ts_ns") >= window_start_ns - margin_ns) & 
        (pl.col("ts_ns") < window_end_ns + margin_ns)
    )
    
    if window_trades.height < 10 or window_snaps.height < 10:
        return None
    
    # Extract numpy arrays for fast processing
    trades_ts = window_trades["ts_ns"].to_numpy()
    trades_price = window_trades["price_ticks"].to_numpy()
    snap_ts = window_snaps["ts_ns"].to_numpy()
    snap_bid = window_snaps["best_bid_ticks"].to_numpy()
    snap_ask = window_snaps["best_ask_ticks"].to_numpy()
    
    # Sort snapshots by timestamp
    sort_idx = np.argsort(snap_ts)
    snap_ts = snap_ts[sort_idx]
    snap_bid = snap_bid[sort_idx]
    snap_ask = snap_ask[sort_idx]
    
    # Find optimal offset
    optimal_offset_ms, outside_rate = find_optimal_offset(
        trades_ts, trades_price, snap_ts, snap_bid, snap_ask,
        search_range_ms, step_ms
    )
    
    return CalibrationResult(
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        optimal_offset_ms=optimal_offset_ms,
        outside_spread_rate=outside_rate,
        trade_count=len(trades_ts),
        snapshot_count=len(snap_ts),
    )


def calibrate_full_day(
    data_root: Path,
    symbol: str,
    date: str,
    window_minutes: int = 10,
    search_range_ms: int = 500,
    step_ms: int = 10,
    output_dir: Optional[Path] = None,
) -> pl.DataFrame:
    """
    Calibrate timestamp offset for a full day using sliding windows.
    
    Returns DataFrame with offset calibration curve.
    """
    logger.info(f"Calibrating {date} with {window_minutes}min windows")
    
    # Load data
    snapshots_lazy, trades_lazy = load_data_lazy(data_root, symbol, date)
    
    # Select only needed columns and collect
    logger.info("Loading snapshot data...")
    snapshots_df = snapshots_lazy.select([
        "ts_ns", "best_bid_ticks", "best_ask_ticks"
    ]).collect()
    logger.info(f"Loaded {snapshots_df.height:,} snapshots")
    
    logger.info("Loading trade data...")
    trades_df = trades_lazy.select([
        "ts_ns", "price_ticks"
    ]).collect()
    logger.info(f"Loaded {trades_df.height:,} trades")
    
    # Get time range
    min_ts = min(snapshots_df["ts_ns"].min(), trades_df["ts_ns"].min())
    max_ts = max(snapshots_df["ts_ns"].max(), trades_df["ts_ns"].max())
    
    # Generate windows
    window_ns = window_minutes * 60 * 1_000_000_000
    results = []
    
    current_start = min_ts
    window_count = 0
    
    while current_start < max_ts:
        window_end = current_start + window_ns
        
        result = calibrate_window(
            trades_df, snapshots_df,
            current_start, window_end,
            search_range_ms, step_ms
        )
        
        if result:
            results.append({
                "window_start_ns": result.window_start_ns,
                "window_end_ns": result.window_end_ns,
                "window_start_time": datetime.fromtimestamp(result.window_start_ns / 1e9).strftime("%Y-%m-%d %H:%M:%S"),
                "optimal_offset_ms": result.optimal_offset_ms,
                "outside_spread_rate": result.outside_spread_rate,
                "trade_count": result.trade_count,
                "snapshot_count": result.snapshot_count,
            })
            window_count += 1
            
            if window_count % 10 == 0:
                logger.info(f"Processed {window_count} windows, current offset: {result.optimal_offset_ms:.0f}ms")
        
        current_start = window_end
    
    # Create result DataFrame
    calibration_df = pl.DataFrame(results)
    
    # Print summary
    if calibration_df.height > 0:
        offsets = calibration_df["optimal_offset_ms"].to_numpy()
        rates = calibration_df["outside_spread_rate"].to_numpy()
        
        print("\n" + "=" * 70)
        print("TIMESTAMP CALIBRATION RESULTS")
        print(f"Date: {date}")
        print(f"Windows: {calibration_df.height} ({window_minutes} min each)")
        print("=" * 70)
        
        print(f"\nOFFSET STATISTICS (ms)")
        print(f"  Mean:   {np.mean(offsets):.1f}ms")
        print(f"  Std:    {np.std(offsets):.1f}ms")
        print(f"  Min:    {np.min(offsets):.0f}ms")
        print(f"  Max:    {np.max(offsets):.0f}ms")
        print(f"  Median: {np.median(offsets):.0f}ms")
        
        print(f"\nOUTSIDE SPREAD RATE (after correction)")
        print(f"  Mean:   {np.mean(rates)*100:.2f}%")
        print(f"  Min:    {np.min(rates)*100:.2f}%")
        print(f"  Max:    {np.max(rates)*100:.2f}%")
        
        # Check for oscillation/patterns
        if len(offsets) > 10:
            diff = np.diff(offsets)
            print(f"\nOFFSET STABILITY")
            print(f"  Mean change between windows: {np.mean(np.abs(diff)):.1f}ms")
            print(f"  Max change between windows:  {np.max(np.abs(diff)):.0f}ms")
            
            # Check for trend
            x = np.arange(len(offsets))
            slope = np.polyfit(x, offsets, 1)[0]
            print(f"  Trend (slope): {slope:.2f}ms per window")
        
        print("=" * 70)
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_dir / f"calibration_{date}.csv"
        calibration_df.write_csv(csv_file)
        logger.info(f"Saved calibration to {csv_file}")
        
        # Also save as parquet for later use
        parquet_file = output_dir / f"calibration_{date}.parquet"
        calibration_df.write_parquet(parquet_file)
        logger.info(f"Saved calibration to {parquet_file}")
    
    return calibration_df


def main():
    parser = argparse.ArgumentParser(description="Calibrate timestamp offset between trades and snapshots")
    parser.add_argument("--data-root", type=Path, default=Path("./data/binance"), help="Data root directory")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--date", type=str, help="Date to calibrate (YYYY-MM-DD)")
    parser.add_argument("--window-minutes", type=int, default=10, help="Window size in minutes")
    parser.add_argument("--search-range-ms", type=int, default=500, help="Search range for offset (±ms)")
    parser.add_argument("--step-ms", type=int, default=10, help="Step size for offset search")
    parser.add_argument("--output-dir", type=Path, default=Path("./data/binance/calibration"), help="Output directory")
    parser.add_argument("--all-dates", action="store_true", help="Calibrate all available dates")
    
    args = parser.parse_args()
    
    if args.all_dates:
        snap_root = args.data_root / "snapshots" / args.symbol
        dates = sorted([d.name for d in snap_root.iterdir() if d.is_dir()])
        logger.info(f"Found {len(dates)} dates: {dates}")
        
        for date in dates:
            try:
                calibrate_full_day(
                    args.data_root, args.symbol, date,
                    args.window_minutes, args.search_range_ms, args.step_ms,
                    args.output_dir
                )
            except Exception as e:
                logger.error(f"Failed to calibrate {date}: {e}")
    elif args.date:
        calibrate_full_day(
            args.data_root, args.symbol, args.date,
            args.window_minutes, args.search_range_ms, args.step_ms,
            args.output_dir
        )
    else:
        parser.error("Either --date or --all-dates is required")


if __name__ == "__main__":
    main()

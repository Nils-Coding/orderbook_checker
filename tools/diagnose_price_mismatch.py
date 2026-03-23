#!/usr/bin/env python3
"""
Diagnose price mismatches between trades and orderbook snapshots.

Analyzes:
- Timestamp-based matching: Find nearest snapshot for each trade
- Penetration check: Is trade within bid-ask spread?
- Volatility correlation: Does deviation correlate with price movement?
- Temporal patterns: When do large deviations occur?
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
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
class MismatchReport:
    """Report of price mismatch analysis."""
    
    date: str
    total_trades: int
    total_snapshots: int
    matched_trades: int
    
    # Deviation statistics
    mean_deviation: float
    std_deviation: float
    max_deviation: int
    min_deviation: int
    median_deviation: float
    
    # Threshold counts
    trades_outside_spread: int
    trades_deviation_gt_10: int
    trades_deviation_gt_100: int
    trades_deviation_gt_500: int
    trades_deviation_gt_1000: int
    
    # Time diff statistics
    mean_time_diff_ms: float
    max_time_diff_ms: float
    
    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print("PRICE MISMATCH DIAGNOSIS REPORT")
        print(f"Date: {self.date}")
        print("=" * 70)
        
        print(f"\nDATA SUMMARY")
        print(f"  Total trades: {self.total_trades:,}")
        print(f"  Total snapshots: {self.total_snapshots:,}")
        print(f"  Matched trades: {self.matched_trades:,}")
        
        print(f"\nDEVIATION STATISTICS (ticks)")
        print(f"  Mean: {self.mean_deviation:.1f}")
        print(f"  Std: {self.std_deviation:.1f}")
        print(f"  Median: {self.median_deviation:.1f}")
        print(f"  Min: {self.min_deviation}")
        print(f"  Max: {self.max_deviation}")
        
        print(f"\nDEVIATION THRESHOLDS")
        print(f"  Trades outside spread: {self.trades_outside_spread:,} ({100*self.trades_outside_spread/self.matched_trades:.2f}%)")
        print(f"  Deviation > 10 ticks: {self.trades_deviation_gt_10:,} ({100*self.trades_deviation_gt_10/self.matched_trades:.2f}%)")
        print(f"  Deviation > 100 ticks: {self.trades_deviation_gt_100:,} ({100*self.trades_deviation_gt_100/self.matched_trades:.2f}%)")
        print(f"  Deviation > 500 ticks: {self.trades_deviation_gt_500:,} ({100*self.trades_deviation_gt_500/self.matched_trades:.2f}%)")
        print(f"  Deviation > 1000 ticks: {self.trades_deviation_gt_1000:,} ({100*self.trades_deviation_gt_1000/self.matched_trades:.2f}%)")
        
        print(f"\nTIME DIFF STATISTICS")
        print(f"  Mean time to nearest snapshot: {self.mean_time_diff_ms:.2f}ms")
        print(f"  Max time to nearest snapshot: {self.max_time_diff_ms:.2f}ms")
        
        print("=" * 70)


def load_data(data_root: Path, symbol: str, date: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load snapshots and trades for a specific date."""
    
    snap_path = data_root / "snapshots" / symbol / date
    trade_path = data_root / "trades" / symbol / date
    
    if not snap_path.exists():
        raise FileNotFoundError(f"Snapshot path not found: {snap_path}")
    if not trade_path.exists():
        raise FileNotFoundError(f"Trade path not found: {trade_path}")
    
    logger.info(f"Loading snapshots from {snap_path}")
    snapshots = pl.read_parquet(f"{snap_path}/**/*.parquet")
    snapshots = snapshots.sort("ts_ns")
    logger.info(f"Loaded {snapshots.height:,} snapshots")
    
    logger.info(f"Loading trades from {trade_path}")
    trades = pl.read_parquet(f"{trade_path}/**/*.parquet")
    trades = trades.sort("ts_ns")
    logger.info(f"Loaded {trades.height:,} trades")
    
    return snapshots, trades


def match_trades_to_snapshots(
    trades: pl.DataFrame,
    snapshots: pl.DataFrame,
    sample_size: Optional[int] = None,
    window_size: int = 1,
) -> pl.DataFrame:
    """
    Match each trade to surrounding snapshots with configurable window.
    
    Uses EXTENDED SPREAD over window: 
    - min(bid_Tn, bid_Tn+1, ..., bid_Tn+window) 
    - max(ask_Tn, ask_Tn+1, ..., ask_Tn+window)
    
    Args:
        window_size: Number of snapshots AFTER Tn to include (default=1 means Tn+Tn+1)
                     Use window_size=10 for 1 second window (10x100ms)
    
    Returns DataFrame with columns:
    - trade_ts_ns, trade_price, snap_ts_ns, snap_bid, snap_ask
    - extended_bid, extended_ask (min/max over window)
    - deviation (0 if within extended spread, otherwise distance to nearest bound)
    - time_diff_ms, window_ms (time span of window used)
    """
    
    if sample_size and sample_size < trades.height:
        logger.info(f"Sampling {sample_size:,} trades for analysis")
        trades = trades.sample(n=sample_size, seed=42)
        trades = trades.sort("ts_ns")
    
    snap_ts = snapshots["ts_ns"].to_numpy()
    snap_bid = snapshots["best_bid_ticks"].to_numpy()
    snap_ask = snapshots["best_ask_ticks"].to_numpy()
    num_snapshots = len(snap_ts)
    
    results = []
    
    trade_count = trades.height
    for i, row in enumerate(trades.iter_rows(named=True)):
        if i % 100000 == 0 and i > 0:
            logger.info(f"Processing trade {i:,}/{trade_count:,}")
        
        trade_ts = row["ts_ns"]
        trade_price = row["price_ticks"]
        
        # Find snapshot BEFORE the trade (Tn)
        idx = np.searchsorted(snap_ts, trade_ts, side="right") - 1
        
        if idx < 0:
            # Trade before first snapshot - skip
            continue
        
        # Tn values
        bid_tn = snap_bid[idx]
        ask_tn = snap_ask[idx]
        snap_time = snap_ts[idx]
        
        # Collect bids/asks over the window [Tn, Tn+1, ..., Tn+window_size]
        end_idx = min(idx + window_size + 1, num_snapshots)
        window_bids = snap_bid[idx:end_idx]
        window_asks = snap_ask[idx:end_idx]
        
        # Extended spread: conservative bounds over entire window
        extended_bid = int(np.min(window_bids))
        extended_ask = int(np.max(window_asks))
        
        # Calculate time span of window
        window_end_time = snap_ts[end_idx - 1] if end_idx > idx else snap_time
        window_ms = (window_end_time - snap_time) / 1_000_000
        
        # Calculate deviation against EXTENDED spread
        if trade_price < extended_bid:
            deviation = trade_price - extended_bid  # negative
        elif trade_price > extended_ask:
            deviation = trade_price - extended_ask  # positive
        else:
            deviation = 0  # within extended spread
        
        time_diff_ms = (trade_ts - snap_time) / 1_000_000
        
        results.append({
            "trade_ts_ns": trade_ts,
            "trade_price": trade_price,
            "snap_ts_ns": snap_time,
            "snap_bid": bid_tn,
            "snap_ask": ask_tn,
            "extended_bid": extended_bid,
            "extended_ask": extended_ask,
            "spread": ask_tn - bid_tn,
            "extended_spread": extended_ask - extended_bid,
            "deviation": deviation,
            "abs_deviation": abs(deviation),
            "time_diff_ms": time_diff_ms,
            "window_ms": window_ms,
        })
    
    return pl.DataFrame(results)


def analyze_by_hour(matches: pl.DataFrame) -> pl.DataFrame:
    """Analyze deviations grouped by hour."""
    
    return matches.with_columns(
        pl.col("trade_ts_ns").map_elements(
            lambda x: datetime.fromtimestamp(x / 1e9).hour,
            return_dtype=pl.Int64
        ).alias("hour")
    ).group_by("hour").agg(
        pl.count().alias("trade_count"),
        pl.col("abs_deviation").mean().alias("mean_deviation"),
        pl.col("abs_deviation").max().alias("max_deviation"),
        pl.col("time_diff_ms").mean().alias("mean_time_diff_ms"),
        (pl.col("abs_deviation") > 100).sum().alias("count_gt_100"),
    ).sort("hour")


def find_large_deviations(
    matches: pl.DataFrame,
    threshold: int = 500,
) -> pl.DataFrame:
    """Find trades with large deviations for detailed inspection."""
    
    large = matches.filter(pl.col("abs_deviation") > threshold)
    
    # Add human-readable timestamp
    large = large.with_columns(
        pl.col("trade_ts_ns").map_elements(
            lambda x: datetime.fromtimestamp(x / 1e9).strftime("%Y-%m-%d %H:%M:%S.%f"),
            return_dtype=pl.String
        ).alias("trade_time")
    )
    
    return large.sort("abs_deviation", descending=True)


def test_window_sizes(
    data_root: Path,
    symbol: str,
    date: str,
    sample_size: int = 10000,
    max_window: int = 10,
) -> None:
    """
    Test different window sizes to find optimal trade-snapshot matching.
    
    This helps identify clock drift between trade and snapshot timestamps.
    """
    snapshots, trades = load_data(data_root, symbol, date)
    
    print("\n" + "=" * 80)
    print("WINDOW SIZE ANALYSIS")
    print(f"Date: {date} | Sample: {sample_size:,} trades | Max window: {max_window}")
    print("=" * 80)
    print(f"\n{'Window':>8} | {'Window ms':>10} | {'Outside %':>10} | {'>100 ticks':>12} | {'>500 ticks':>12}")
    print("-" * 70)
    
    results = []
    
    for window in range(0, max_window + 1):
        matches = match_trades_to_snapshots(trades, snapshots, sample_size, window_size=window)
        abs_dev = matches["abs_deviation"].to_numpy()
        
        outside_pct = 100 * (abs_dev > 0).sum() / len(abs_dev)
        gt_100_pct = 100 * (abs_dev > 100).sum() / len(abs_dev)
        gt_500_pct = 100 * (abs_dev > 500).sum() / len(abs_dev)
        
        # Approximate window time (100ms per snapshot)
        window_time_ms = window * 100
        
        print(f"{window:>8} | {window_time_ms:>10}ms | {outside_pct:>9.2f}% | {gt_100_pct:>11.2f}% | {gt_500_pct:>11.2f}%")
        
        results.append({
            "window": window,
            "window_ms": window_time_ms,
            "outside_pct": outside_pct,
            "gt_100_pct": gt_100_pct,
            "gt_500_pct": gt_500_pct,
        })
    
    print("-" * 70)
    
    # Find optimal window (lowest outside_pct)
    best = min(results, key=lambda x: x["outside_pct"])
    print(f"\nBEST WINDOW: {best['window']} snapshots ({best['window_ms']}ms)")
    print(f"  -> {best['outside_pct']:.2f}% trades outside spread")
    
    if best["window"] > 1:
        print(f"\n>>> INTERPRETATION: Clock drift of approximately {best['window'] * 50}ms detected")
        print(f"    (Window {best['window']} = {best['window_ms']}ms provides best match)")


def diagnose(
    data_root: Path,
    symbol: str,
    date: str,
    sample_size: Optional[int] = None,
    output_dir: Optional[Path] = None,
    window_size: int = 1,
) -> MismatchReport:
    """Run full price mismatch diagnosis."""
    
    snapshots, trades = load_data(data_root, symbol, date)
    
    logger.info(f"Matching trades to snapshots (window={window_size})...")
    matches = match_trades_to_snapshots(trades, snapshots, sample_size, window_size=window_size)
    logger.info(f"Matched {matches.height:,} trades")
    
    # Calculate statistics
    deviations = matches["deviation"].to_numpy()
    abs_deviations = matches["abs_deviation"].to_numpy()
    time_diffs = matches["time_diff_ms"].to_numpy()
    
    report = MismatchReport(
        date=date,
        total_trades=trades.height,
        total_snapshots=snapshots.height,
        matched_trades=matches.height,
        mean_deviation=float(np.mean(abs_deviations)),
        std_deviation=float(np.std(abs_deviations)),
        max_deviation=int(np.max(abs_deviations)),
        min_deviation=int(np.min(deviations)),
        median_deviation=float(np.median(abs_deviations)),
        trades_outside_spread=int((abs_deviations > 0).sum()),
        trades_deviation_gt_10=int((abs_deviations > 10).sum()),
        trades_deviation_gt_100=int((abs_deviations > 100).sum()),
        trades_deviation_gt_500=int((abs_deviations > 500).sum()),
        trades_deviation_gt_1000=int((abs_deviations > 1000).sum()),
        mean_time_diff_ms=float(np.mean(time_diffs)),
        max_time_diff_ms=float(np.max(time_diffs)),
    )
    
    report.print_report()
    
    # Hourly analysis
    logger.info("Analyzing by hour...")
    hourly = analyze_by_hour(matches)
    print("\nHOURLY BREAKDOWN")
    print(hourly)
    
    # Find large deviations
    logger.info("Finding large deviations...")
    large = find_large_deviations(matches, threshold=500)
    if large.height > 0:
        print(f"\nLARGE DEVIATIONS (>{500} ticks): {large.height} trades")
        print(large.select(["trade_time", "trade_price", "snap_bid", "snap_ask", "deviation", "time_diff_ms"]).head(20))
    
    # Save outputs
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all matches
        matches_file = output_dir / f"matches_{date}.parquet"
        matches.write_parquet(matches_file)
        logger.info(f"Saved matches to {matches_file}")
        
        # Save large deviations
        if large.height > 0:
            large_file = output_dir / f"large_deviations_{date}.csv"
            large.write_csv(large_file)
            logger.info(f"Saved large deviations to {large_file}")
        
        # Save hourly analysis
        hourly_file = output_dir / f"hourly_{date}.csv"
        hourly.write_csv(hourly_file)
        logger.info(f"Saved hourly analysis to {hourly_file}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Diagnose price mismatches between trades and snapshots")
    parser.add_argument("--data-root", type=Path, default=Path("./data/binance"), help="Data root directory")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--date", type=str, required=True, help="Date to analyze (YYYY-MM-DD)")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (for faster analysis)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for results")
    parser.add_argument("--all-dates", action="store_true", help="Analyze all available dates")
    parser.add_argument("--window", type=int, default=1, help="Window size: snapshots after Tn to include (default=1)")
    parser.add_argument("--test-windows", action="store_true", help="Test different window sizes (0-10)")
    
    args = parser.parse_args()
    
    if args.test_windows:
        # Run window size analysis
        test_window_sizes(
            args.data_root, args.symbol, args.date,
            sample_size=args.sample or 10000,
            max_window=10,
        )
        return
    
    if args.all_dates:
        # Find all available dates
        snap_root = args.data_root / "snapshots" / args.symbol
        dates = sorted([d.name for d in snap_root.iterdir() if d.is_dir()])
        logger.info(f"Found {len(dates)} dates: {dates}")
        
        for date in dates:
            try:
                diagnose(args.data_root, args.symbol, date, args.sample, args.output_dir, args.window)
            except Exception as e:
                logger.error(f"Failed to analyze {date}: {e}")
    else:
        diagnose(args.data_root, args.symbol, args.date, args.sample, args.output_dir, args.window)


if __name__ == "__main__":
    main()

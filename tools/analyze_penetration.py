#!/usr/bin/env python3
"""
Trade Penetration Analysis Tool

Analyzes how deep trades penetrate into the orderbook by matching
trades to the nearest previous snapshot and calculating which
level would be consumed.

Usage:
    python tools/analyze_penetration.py --data_root ./data --symbol BTCUSDT --out ./reports
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_snapshots(data_root: Path, symbol: str) -> pd.DataFrame:
    """Load all snapshot parquet files for a symbol."""
    snap_dir = data_root / "snapshots" / f"symbol={symbol}"

    if not snap_dir.exists():
        logger.error(f"Snapshot directory not found: {snap_dir}")
        return pd.DataFrame()

    files = sorted(snap_dir.rglob("*.parquet"))
    if not files:
        logger.error(f"No snapshot files found in {snap_dir}")
        return pd.DataFrame()

    logger.info(f"Loading {len(files)} snapshot files...")

    dfs = []
    for f in files:
        try:
            # Use ParquetFile.read() to avoid pyarrow.dataset schema merge issues
            # (observed as string vs dictionary merge errors even for single files).
            tbl = pq.ParquetFile(f).read()
            df = tbl.to_pandas()
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("ts_ns").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} snapshots")
    return df


def load_trades(data_root: Path, symbol: str) -> pd.DataFrame:
    """Load all trade parquet files for a symbol."""
    trade_dir = data_root / "trades" / f"symbol={symbol}"

    if not trade_dir.exists():
        logger.error(f"Trade directory not found: {trade_dir}")
        return pd.DataFrame()

    files = sorted(trade_dir.rglob("*.parquet"))
    if not files:
        logger.error(f"No trade files found in {trade_dir}")
        return pd.DataFrame()

    logger.info(f"Loading {len(files)} trade files...")

    dfs = []
    for f in files:
        try:
            # Use ParquetFile.read() to avoid pyarrow.dataset schema merge issues
            # (observed as string vs dictionary merge errors even for single files).
            tbl = pq.ParquetFile(f).read()

            df = tbl.to_pandas()
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("ts_ns").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} trades")
    return df


def calculate_penetration(
    trade_qty_lots: int,
    orderbook_qty_lots: np.ndarray,
    max_level: int = 1000,
) -> int:
    """
    Calculate which level a trade would penetrate to.

    Args:
        trade_qty_lots: Trade quantity in lots
        orderbook_qty_lots: Array of quantities at each level (1000 levels)
        max_level: Maximum level to check

    Returns:
        Level number (1-1000) or 1001 for overflow
    """
    cum_qty = 0
    for level, qty in enumerate(orderbook_qty_lots[:max_level], start=1):
        cum_qty += qty
        if cum_qty >= trade_qty_lots:
            return level
    return max_level + 1  # Overflow


def analyze_trades(
    snapshots: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match trades to snapshots and calculate penetration.

    Uses merge_asof to find the nearest previous snapshot for each trade.

    Args:
        snapshots: DataFrame with orderbook snapshots
        trades: DataFrame with trades

    Returns:
        DataFrame with trade analysis results
    """
    if snapshots.empty or trades.empty:
        logger.error("Empty snapshots or trades")
        return pd.DataFrame()

    logger.info("Matching trades to snapshots...")

    # Ensure sorted by timestamp
    snapshots = snapshots.sort_values("ts_ns").reset_index(drop=True)
    trades = trades.sort_values("ts_ns").reset_index(drop=True)

    # Use merge_asof to find nearest previous snapshot
    # direction='backward' ensures ts_snapshot <= ts_trade
    merged = pd.merge_asof(
        trades,
        snapshots[["ts_ns", "best_bid_ticks", "best_ask_ticks", "bids_qty_lots", "asks_qty_lots"]],
        on="ts_ns",
        direction="backward",
        suffixes=("", "_snap"),
    )

    logger.info(f"Matched {len(merged)} trades to snapshots")

    # Calculate penetration for each trade
    results = []
    skipped_no_snapshot = 0
    for idx, row in merged.iterrows():
        # No matching snapshot found: merge_asof fills snapshot columns with NaN (scalar),
        # but the column itself typically contains list/array values when present.
        bids_val = row.get("bids_qty_lots")
        if bids_val is None or (pd.api.types.is_scalar(bids_val) and pd.isna(bids_val)):
            skipped_no_snapshot += 1
            continue

        # Determine which side based on is_buyer_maker
        # is_buyer_maker=True -> taker is seller, consumes bids
        # is_buyer_maker=False -> taker is buyer, consumes asks
        if row["is_buyer_maker"]:
            qty_levels = np.array(row["bids_qty_lots"])
        else:
            qty_levels = np.array(row["asks_qty_lots"])

        penetration = calculate_penetration(row["qty_lots"], qty_levels)

        results.append({
            "ts_ns": row["ts_ns"],
            "symbol": row["symbol"],
            "price_ticks": row["price_ticks"],
            "qty_lots": row["qty_lots"],
            "is_buyer_maker": row["is_buyer_maker"],
            "side": "bid" if row["is_buyer_maker"] else "ask",
            "penetration_level": penetration,
            "is_overflow": penetration > 1000,
        })

    result_df = pd.DataFrame(results)
    logger.info(f"Calculated penetration for {len(result_df)} trades")

    return result_df


def generate_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Generate penetration summary statistics.

    Returns breakdown by threshold levels (20, 50, 100, 200, 500, 1000).
    """
    if analysis.empty:
        return pd.DataFrame()

    thresholds = [20, 50, 100, 200, 500, 1000]
    total_trades = len(analysis)
    total_notional = analysis["qty_lots"].sum()

    summary_rows = []

    for threshold in thresholds:
        exceeds = analysis[analysis["penetration_level"] > threshold]
        trades_exceeding = len(exceeds)
        notional_exceeding = exceeds["qty_lots"].sum()

        summary_rows.append({
            "threshold": threshold,
            "trades_exceeding": trades_exceeding,
            "trades_exceeding_pct": (trades_exceeding / total_trades * 100) if total_trades > 0 else 0,
            "notional_exceeding": notional_exceeding,
            "notional_exceeding_pct": (notional_exceeding / total_notional * 100) if total_notional > 0 else 0,
        })

    # Add overflow stats
    overflow = analysis[analysis["is_overflow"]]
    summary_rows.append({
        "threshold": 1001,
        "trades_exceeding": len(overflow),
        "trades_exceeding_pct": (len(overflow) / total_trades * 100) if total_trades > 0 else 0,
        "notional_exceeding": overflow["qty_lots"].sum(),
        "notional_exceeding_pct": (overflow["qty_lots"].sum() / total_notional * 100) if total_notional > 0 else 0,
    })

    return pd.DataFrame(summary_rows)


def generate_side_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Generate summary broken down by side (bid/ask)."""
    if analysis.empty:
        return pd.DataFrame()

    summaries = []
    for side in ["bid", "ask"]:
        side_data = analysis[analysis["side"] == side]
        if side_data.empty:
            continue

        summary = generate_summary(side_data)
        summary["side"] = side
        summaries.append(summary)

    if summaries:
        return pd.concat(summaries, ignore_index=True)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trade penetration into orderbook levels"
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Root directory containing snapshots/ and trades/ subdirectories",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., BTCUSDT)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./reports"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also output CSV files",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.data_root.exists():
        logger.error(f"Data root does not exist: {args.data_root}")
        sys.exit(1)

    # Load data
    snapshots = load_snapshots(args.data_root, args.symbol)
    trades = load_trades(args.data_root, args.symbol)

    if snapshots.empty or trades.empty:
        logger.error("Failed to load data")
        sys.exit(1)

    # Analyze
    analysis = analyze_trades(snapshots, trades)
    if analysis.empty:
        logger.error("Analysis produced no results")
        sys.exit(1)

    # Generate summaries
    summary = generate_summary(analysis)
    side_summary = generate_side_summary(analysis)

    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)

    # Write results
    analysis_file = args.out / "trade_analysis.parquet"
    summary_file = args.out / "penetration_summary.parquet"
    side_summary_file = args.out / "penetration_by_side.parquet"

    analysis.to_parquet(analysis_file, compression="zstd")
    summary.to_parquet(summary_file, compression="zstd")
    if not side_summary.empty:
        side_summary.to_parquet(side_summary_file, compression="zstd")

    logger.info(f"Wrote analysis to {analysis_file}")
    logger.info(f"Wrote summary to {summary_file}")

    if args.csv:
        analysis.to_csv(args.out / "trade_analysis.csv", index=False)
        summary.to_csv(args.out / "penetration_summary.csv", index=False)
        if not side_summary.empty:
            side_summary.to_csv(args.out / "penetration_by_side.csv", index=False)
        logger.info("Also wrote CSV files")

    # Print summary
    print("\n" + "=" * 60)
    print("PENETRATION SUMMARY")
    print("=" * 60)
    print(f"\nSymbol: {args.symbol}")
    print(f"Total trades analyzed: {len(analysis)}")
    print(f"Total notional (lots): {analysis['qty_lots'].sum()}")
    print(f"\nTrades exceeding level thresholds:")
    print(summary.to_string(index=False))

    if not side_summary.empty:
        print("\n" + "-" * 60)
        print("BY SIDE:")
        print(side_summary.to_string(index=False))

    # Key insights
    p50 = analysis["penetration_level"].quantile(0.5)
    p90 = analysis["penetration_level"].quantile(0.9)
    p99 = analysis["penetration_level"].quantile(0.99)
    max_pen = analysis["penetration_level"].max()

    print("\n" + "-" * 60)
    print("PENETRATION PERCENTILES:")
    print(f"  50th percentile: Level {p50:.0f}")
    print(f"  90th percentile: Level {p90:.0f}")
    print(f"  99th percentile: Level {p99:.0f}")
    print(f"  Maximum: Level {max_pen}")

    overflow_pct = (analysis["is_overflow"].sum() / len(analysis)) * 100
    print(f"\nOverflow (>1000 levels): {overflow_pct:.2f}%")

    print("=" * 60)

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()


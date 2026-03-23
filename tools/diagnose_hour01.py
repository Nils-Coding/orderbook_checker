#!/usr/bin/env python3
"""
Diagnose hour=01 file count anomaly.

Analyzes:
- File counts per hour (snapshots and trades)
- Rows per file comparison
- Timestamp gaps and overlaps
- Part number sequence analysis
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class HourAnalysis:
    """Analysis results for a single hour."""
    
    date: str
    hour: int
    data_type: str  # "snapshots" or "trades"
    
    file_count: int
    total_rows: int
    rows_per_file_avg: float
    rows_per_file_min: int
    rows_per_file_max: int
    
    first_timestamp: str
    last_timestamp: str
    duration_minutes: float
    
    gap_count: int
    max_gap_ms: float
    overlap_count: int
    
    part_numbers: list[int]
    missing_parts: list[int]
    extra_parts: list[int]


def analyze_hour(
    data_root: Path,
    symbol: str,
    date: str,
    hour: int,
    data_type: str,
) -> Optional[HourAnalysis]:
    """Analyze a single hour of data."""
    
    hour_str = f"hour={hour:02d}"
    path = data_root / data_type / symbol / date / hour_str
    
    if not path.exists():
        return None
    
    files = sorted(path.glob("*.parquet"))
    if not files:
        return None
    
    # Extract part numbers
    part_numbers = []
    for f in files:
        # part-0000.parquet -> 0
        try:
            part_num = int(f.stem.split("-")[1])
            part_numbers.append(part_num)
        except (IndexError, ValueError):
            pass
    
    # Find missing and extra parts
    if part_numbers:
        expected_parts = set(range(max(part_numbers) + 1))
        actual_parts = set(part_numbers)
        missing_parts = sorted(expected_parts - actual_parts)
        # Extra parts would be any beyond expected ~60
        extra_parts = [p for p in part_numbers if p >= 60]
    else:
        missing_parts = []
        extra_parts = []
    
    # Load all files and collect stats
    file_stats = []
    all_data = []
    
    for f in files:
        try:
            df = pl.read_parquet(f)
            file_stats.append({
                "file": f.name,
                "rows": df.height,
            })
            all_data.append(df.select(["ts_ns"]))
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")
    
    if not all_data:
        return None
    
    # Combine all timestamps
    combined = pl.concat(all_data).sort("ts_ns")
    timestamps = combined["ts_ns"].to_numpy()
    
    # Calculate gaps
    if len(timestamps) > 1:
        import numpy as np
        deltas_ms = np.diff(timestamps) / 1_000_000
        
        # For snapshots, expect ~100ms intervals; for trades, variable
        if data_type == "snapshots":
            gap_threshold_ms = 200  # Gaps > 200ms are significant
        else:
            gap_threshold_ms = 1000  # Gaps > 1s for trades
        
        gaps = deltas_ms[deltas_ms > gap_threshold_ms]
        gap_count = len(gaps)
        max_gap_ms = float(np.max(deltas_ms)) if len(deltas_ms) > 0 else 0
        
        # Check for overlaps (negative deltas shouldn't exist in sorted data)
        overlap_count = int((deltas_ms < 0).sum())
    else:
        gap_count = 0
        max_gap_ms = 0
        overlap_count = 0
    
    # Timestamps
    first_ts = datetime.fromtimestamp(timestamps[0] / 1e9)
    last_ts = datetime.fromtimestamp(timestamps[-1] / 1e9)
    duration_minutes = (timestamps[-1] - timestamps[0]) / 1e9 / 60
    
    # File stats
    rows_list = [s["rows"] for s in file_stats]
    
    return HourAnalysis(
        date=date,
        hour=hour,
        data_type=data_type,
        file_count=len(files),
        total_rows=sum(rows_list),
        rows_per_file_avg=sum(rows_list) / len(rows_list) if rows_list else 0,
        rows_per_file_min=min(rows_list) if rows_list else 0,
        rows_per_file_max=max(rows_list) if rows_list else 0,
        first_timestamp=first_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
        last_timestamp=last_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
        duration_minutes=duration_minutes,
        gap_count=gap_count,
        max_gap_ms=max_gap_ms,
        overlap_count=overlap_count,
        part_numbers=part_numbers,
        missing_parts=missing_parts,
        extra_parts=extra_parts,
    )


def compare_hours(
    data_root: Path,
    symbol: str,
    date: str,
    data_type: str,
) -> pl.DataFrame:
    """Compare all hours for a given date and data type."""
    
    results = []
    
    for hour in range(24):
        analysis = analyze_hour(data_root, symbol, date, hour, data_type)
        if analysis:
            results.append({
                "hour": hour,
                "files": analysis.file_count,
                "rows": analysis.total_rows,
                "rows_per_file": round(analysis.rows_per_file_avg, 1),
                "min_rows": analysis.rows_per_file_min,
                "max_rows": analysis.rows_per_file_max,
                "gaps": analysis.gap_count,
                "max_gap_ms": round(analysis.max_gap_ms, 1),
                "extra_parts": len(analysis.extra_parts),
                "duration_min": round(analysis.duration_minutes, 1),
            })
    
    return pl.DataFrame(results)


def diagnose_hour01(
    data_root: Path,
    symbol: str,
    dates: list[str],
    output_dir: Optional[Path] = None,
) -> None:
    """Run full hour=01 diagnosis across multiple dates."""
    
    print("\n" + "=" * 80)
    print("HOUR=01 ANOMALY DIAGNOSIS")
    print("=" * 80)
    
    all_results = []
    
    for date in dates:
        print(f"\n{'=' * 40}")
        print(f"DATE: {date}")
        print("=" * 40)
        
        # Analyze hour=01 specifically
        snap_h01 = analyze_hour(data_root, symbol, date, 1, "snapshots")
        trade_h01 = analyze_hour(data_root, symbol, date, 1, "trades")
        
        # Also analyze hour=02 as reference
        snap_h02 = analyze_hour(data_root, symbol, date, 2, "snapshots")
        trade_h02 = analyze_hour(data_root, symbol, date, 2, "trades")
        
        print("\nHOUR=01 vs HOUR=02 COMPARISON:")
        print("-" * 60)
        
        if snap_h01 and snap_h02:
            print(f"\nSNAPSHOTS:")
            print(f"  hour=01: {snap_h01.file_count} files, {snap_h01.total_rows:,} rows, {snap_h01.rows_per_file_avg:.1f} rows/file")
            print(f"  hour=02: {snap_h02.file_count} files, {snap_h02.total_rows:,} rows, {snap_h02.rows_per_file_avg:.1f} rows/file")
            print(f"  Ratio: {snap_h01.file_count / snap_h02.file_count:.2f}x files")
            
            if snap_h01.extra_parts:
                print(f"  Extra parts (>=60): {snap_h01.extra_parts}")
            if snap_h01.missing_parts:
                print(f"  Missing parts: {snap_h01.missing_parts[:10]}..." if len(snap_h01.missing_parts) > 10 else f"  Missing parts: {snap_h01.missing_parts}")
        
        if trade_h01 and trade_h02:
            print(f"\nTRADES:")
            print(f"  hour=01: {trade_h01.file_count} files, {trade_h01.total_rows:,} rows, {trade_h01.rows_per_file_avg:.1f} rows/file")
            print(f"  hour=02: {trade_h02.file_count} files, {trade_h02.total_rows:,} rows, {trade_h02.rows_per_file_avg:.1f} rows/file")
            print(f"  Ratio: {trade_h01.file_count / trade_h02.file_count:.2f}x files")
            
            if trade_h01.extra_parts:
                print(f"  Extra parts (>=60): {len(trade_h01.extra_parts)} parts")
        
        # Timestamp analysis
        print(f"\nTIMESTAMP RANGE:")
        if snap_h01:
            print(f"  Snapshots: {snap_h01.first_timestamp} to {snap_h01.last_timestamp}")
            print(f"             Duration: {snap_h01.duration_minutes:.1f} minutes, Gaps: {snap_h01.gap_count}")
        if trade_h01:
            print(f"  Trades:    {trade_h01.first_timestamp} to {trade_h01.last_timestamp}")
            print(f"             Duration: {trade_h01.duration_minutes:.1f} minutes, Max gap: {trade_h01.max_gap_ms:.0f}ms")
        
        # Store for summary
        all_results.append({
            "date": date,
            "snap_files_h01": snap_h01.file_count if snap_h01 else 0,
            "snap_files_h02": snap_h02.file_count if snap_h02 else 0,
            "trade_files_h01": trade_h01.file_count if trade_h01 else 0,
            "trade_files_h02": trade_h02.file_count if trade_h02 else 0,
            "snap_rows_h01": snap_h01.total_rows if snap_h01 else 0,
            "trade_rows_h01": trade_h01.total_rows if trade_h01 else 0,
            "snap_extra_parts": len(snap_h01.extra_parts) if snap_h01 else 0,
            "trade_rows_per_file": trade_h01.rows_per_file_avg if trade_h01 else 0,
        })
        
        # Full hourly comparison
        print(f"\nALL HOURS - SNAPSHOTS:")
        snap_hourly = compare_hours(data_root, symbol, date, "snapshots")
        print(snap_hourly)
        
        print(f"\nALL HOURS - TRADES:")
        trade_hourly = compare_hours(data_root, symbol, date, "trades")
        print(trade_hourly)
    
    # Summary across all dates
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL DATES")
    print("=" * 80)
    summary = pl.DataFrame(all_results)
    print(summary)
    
    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    # Check if trade files are consistently high
    avg_trade_files_h01 = summary["trade_files_h01"].mean()
    avg_trade_files_h02 = summary["trade_files_h02"].mean()
    
    print(f"\nAvg trade files in hour=01: {avg_trade_files_h01:.0f}")
    print(f"Avg trade files in hour=02: {avg_trade_files_h02:.0f}")
    print(f"Ratio: {avg_trade_files_h01 / avg_trade_files_h02:.1f}x")
    
    avg_rows_per_file = summary["trade_rows_per_file"].mean()
    print(f"\nAvg rows per trade file in hour=01: {avg_rows_per_file:.1f}")
    print(f"Expected rows per file: ~1000-3000 (60s chunks with ~20-50 trades/s)")
    
    if avg_trade_files_h01 > avg_trade_files_h02 * 5:
        print("\n>>> ISSUE: Trade files in hour=01 are significantly fragmented")
        print(">>> Likely cause: Writer was restarted frequently or had memory issues")
        print(">>> The daily_report cronjob at 01:00 UTC may be causing this")
    
    # Save outputs
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / "hour01_summary.csv"
        summary.write_csv(summary_file)
        logger.info(f"Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose hour=01 file anomaly")
    parser.add_argument("--data-root", type=Path, default=Path("./data/binance"), help="Data root directory")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--dates", type=str, nargs="+", help="Dates to analyze (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    if not args.dates:
        # Find all available dates
        snap_root = args.data_root / "snapshots" / args.symbol
        if snap_root.exists():
            args.dates = sorted([d.name for d in snap_root.iterdir() if d.is_dir()])
    
    if not args.dates:
        logger.error("No dates found or specified")
        return
    
    logger.info(f"Analyzing dates: {args.dates}")
    diagnose_hour01(args.data_root, args.symbol, args.dates, args.output_dir)


if __name__ == "__main__":
    main()

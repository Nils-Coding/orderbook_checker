#!/usr/bin/env python3
"""
Heal snapshot timestamps using calibration data.

Applies the calibrated timestamp offsets to create corrected snapshot files.
Original files are NOT modified - new files are created in a separate directory.

The offset is interpolated between calibration windows for smooth correction.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_calibration(calibration_path: Path) -> pl.DataFrame:
    """Load calibration data from parquet or CSV."""
    
    if calibration_path.suffix == ".parquet":
        return pl.read_parquet(calibration_path)
    elif calibration_path.suffix == ".csv":
        return pl.read_csv(calibration_path)
    else:
        raise ValueError(f"Unsupported calibration file format: {calibration_path.suffix}")


def create_offset_interpolator(calibration_df: pl.DataFrame):
    """
    Create an interpolation function for timestamp offsets.
    
    Returns a function that takes a timestamp (ns) and returns the offset (ns).
    """
    # Get window centers and offsets
    window_starts = calibration_df["window_start_ns"].to_numpy()
    window_ends = calibration_df["window_end_ns"].to_numpy()
    offsets_ms = calibration_df["optimal_offset_ms"].to_numpy()
    
    # Use window centers for interpolation
    window_centers = (window_starts + window_ends) // 2
    offsets_ns = offsets_ms * 1_000_000
    
    def interpolate_offset(ts_ns: np.ndarray) -> np.ndarray:
        """Interpolate offset for given timestamps."""
        return np.interp(ts_ns, window_centers, offsets_ns)
    
    return interpolate_offset


def heal_snapshots_for_date(
    data_root: Path,
    symbol: str,
    date: str,
    calibration_path: Path,
    output_root: Path,
    verify: bool = True,
) -> dict:
    """
    Apply timestamp correction to snapshot files for a single date.
    
    Args:
        data_root: Original data root
        symbol: Trading symbol
        date: Date to process
        calibration_path: Path to calibration file
        output_root: Root directory for healed data
        verify: If True, run verification after healing
    
    Returns:
        Statistics about the healing process
    """
    snap_path = data_root / "snapshots" / symbol / date
    output_path = output_root / "snapshots" / symbol / date
    
    if not snap_path.exists():
        raise FileNotFoundError(f"Snapshot path not found: {snap_path}")
    
    # Load calibration
    logger.info(f"Loading calibration from {calibration_path}")
    calibration_df = load_calibration(calibration_path)
    interpolate_offset = create_offset_interpolator(calibration_df)
    
    # Get calibration stats
    offsets_ms = calibration_df["optimal_offset_ms"].to_numpy()
    logger.info(f"Calibration: mean offset {np.mean(offsets_ms):.1f}ms, std {np.std(offsets_ms):.1f}ms")
    
    # Process each hour
    hour_dirs = sorted([d for d in snap_path.iterdir() if d.is_dir() and d.name.startswith("hour=")])
    
    total_snapshots = 0
    total_files = 0
    
    for hour_dir in hour_dirs:
        hour = hour_dir.name  # e.g., "hour=00"
        output_hour_dir = output_path / hour
        output_hour_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_files = sorted(hour_dir.glob("*.parquet"))
        
        for parquet_file in parquet_files:
            # Read original file
            df = pl.read_parquet(parquet_file)
            
            if df.height == 0:
                continue
            
            # Get timestamps
            ts_ns = df["ts_ns"].to_numpy()
            
            # Calculate offsets for each timestamp
            offsets_ns = interpolate_offset(ts_ns)
            
            # Apply correction: ADD offset to snapshot timestamps
            # (positive offset means snapshot was behind, so we shift forward)
            corrected_ts_ns = ts_ns + offsets_ns.astype(np.int64)
            
            # Create corrected DataFrame
            corrected_df = df.with_columns(
                pl.Series("ts_ns", corrected_ts_ns),
                pl.Series("ts_ns_original", ts_ns),  # Keep original for reference
                pl.Series("offset_applied_ns", offsets_ns.astype(np.int64)),
            )
            
            # Write to output
            output_file = output_hour_dir / parquet_file.name
            corrected_df.write_parquet(output_file)
            
            total_snapshots += df.height
            total_files += 1
        
        logger.info(f"Processed {hour}: {len(parquet_files)} files")
    
    stats = {
        "date": date,
        "total_files": total_files,
        "total_snapshots": total_snapshots,
        "output_path": str(output_path),
    }
    
    # Verification
    if verify:
        logger.info("Running verification...")
        verify_stats = verify_healing(data_root, output_root, symbol, date)
        stats.update(verify_stats)
    
    return stats


def verify_healing(
    original_root: Path,
    healed_root: Path,
    symbol: str,
    date: str,
    sample_size: int = 10000,
) -> dict:
    """
    Verify that healing improved the trade-snapshot alignment.
    
    Compares "outside spread" rate before and after healing.
    """
    from tools.diagnose_price_mismatch import match_trades_to_snapshots, load_data
    
    # Load original data
    logger.info("Loading original snapshots...")
    original_snaps = pl.read_parquet(f"{original_root}/snapshots/{symbol}/{date}/**/*.parquet")
    original_snaps = original_snaps.sort("ts_ns")
    
    # Load healed data
    logger.info("Loading healed snapshots...")
    healed_snaps = pl.read_parquet(f"{healed_root}/snapshots/{symbol}/{date}/**/*.parquet")
    healed_snaps = healed_snaps.sort("ts_ns")
    
    # Load trades
    logger.info("Loading trades...")
    trades = pl.read_parquet(f"{original_root}/trades/{symbol}/{date}/**/*.parquet")
    trades = trades.sort("ts_ns")
    
    # Sample for faster verification
    if trades.height > sample_size:
        trades = trades.sample(n=sample_size, seed=42).sort("ts_ns")
    
    # Calculate outside rate for original
    logger.info("Calculating original outside-spread rate...")
    original_matches = match_trades_to_snapshots(trades, original_snaps, window_size=0)
    original_outside_rate = (original_matches["abs_deviation"].to_numpy() > 0).mean()
    
    # Calculate outside rate for healed
    logger.info("Calculating healed outside-spread rate...")
    healed_matches = match_trades_to_snapshots(trades, healed_snaps, window_size=0)
    healed_outside_rate = (healed_matches["abs_deviation"].to_numpy() > 0).mean()
    
    improvement = original_outside_rate - healed_outside_rate
    
    print("\n" + "=" * 70)
    print("HEALING VERIFICATION")
    print("=" * 70)
    print(f"  Original outside-spread rate: {original_outside_rate*100:.2f}%")
    print(f"  Healed outside-spread rate:   {healed_outside_rate*100:.2f}%")
    print(f"  Improvement:                  {improvement*100:.2f}% points")
    print("=" * 70)
    
    return {
        "original_outside_rate": original_outside_rate,
        "healed_outside_rate": healed_outside_rate,
        "improvement": improvement,
    }


def heal_all_dates(
    data_root: Path,
    symbol: str,
    calibration_dir: Path,
    output_root: Path,
    verify: bool = True,
) -> None:
    """Heal all dates that have calibration data."""
    
    # Find all calibration files
    calibration_files = sorted(calibration_dir.glob("calibration_*.parquet"))
    
    if not calibration_files:
        logger.error(f"No calibration files found in {calibration_dir}")
        return
    
    logger.info(f"Found {len(calibration_files)} calibration files")
    
    all_stats = []
    
    for cal_file in calibration_files:
        # Extract date from filename: calibration_2026-03-15.parquet
        date = cal_file.stem.replace("calibration_", "")
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {date}")
        logger.info(f"{'='*50}")
        
        try:
            stats = heal_snapshots_for_date(
                data_root, symbol, date, cal_file, output_root, verify
            )
            all_stats.append(stats)
            
            logger.info(f"Completed {date}: {stats['total_snapshots']:,} snapshots healed")
            if verify and "improvement" in stats:
                logger.info(f"  Improvement: {stats['improvement']*100:.2f}% points")
        
        except Exception as e:
            logger.error(f"Failed to process {date}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("HEALING SUMMARY")
    print("=" * 70)
    
    total_snapshots = sum(s.get("total_snapshots", 0) for s in all_stats)
    total_files = sum(s.get("total_files", 0) for s in all_stats)
    
    print(f"  Total dates processed: {len(all_stats)}")
    print(f"  Total files created:   {total_files:,}")
    print(f"  Total snapshots:       {total_snapshots:,}")
    
    if verify:
        improvements = [s.get("improvement", 0) for s in all_stats if "improvement" in s]
        if improvements:
            print(f"  Avg improvement:       {np.mean(improvements)*100:.2f}% points")
    
    print(f"\nHealed data saved to: {output_root}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Heal snapshot timestamps using calibration data")
    parser.add_argument("--data-root", type=Path, default=Path("./data/binance"), help="Original data root")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--date", type=str, help="Single date to heal (YYYY-MM-DD)")
    parser.add_argument("--calibration", type=Path, help="Calibration file (for single date)")
    parser.add_argument("--calibration-dir", type=Path, default=Path("./data/binance/calibration"), 
                        help="Calibration directory (for all dates)")
    parser.add_argument("--output-root", type=Path, default=Path("./data/binance_healed"), 
                        help="Output root for healed data")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    parser.add_argument("--all-dates", action="store_true", help="Heal all dates with calibration")
    
    args = parser.parse_args()
    
    if args.all_dates:
        heal_all_dates(
            args.data_root, args.symbol, args.calibration_dir, 
            args.output_root, verify=not args.no_verify
        )
    elif args.date and args.calibration:
        stats = heal_snapshots_for_date(
            args.data_root, args.symbol, args.date, args.calibration,
            args.output_root, verify=not args.no_verify
        )
        print(f"\nHealing complete: {stats['total_snapshots']:,} snapshots processed")
        print(f"Output: {stats['output_path']}")
    else:
        parser.error("Either --all-dates or both --date and --calibration are required")


if __name__ == "__main__":
    main()

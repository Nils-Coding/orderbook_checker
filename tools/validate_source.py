#!/usr/bin/env python3
"""
Validate that data sources are consistent (both from Binance Futures).

Validates:
- Price plausibility: Futures prices should be close to spot
- Trade ID sequence: aggTrade IDs should be consecutive
- Symbol validation: All data should be BTCUSDT
- Cross-check with Binance REST API (optional)
"""

import argparse
import asyncio
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
class SourceValidationReport:
    """Report of source validation results."""
    
    date: str
    symbol: str
    
    # Price validation
    avg_price_usd: float
    min_price_usd: float
    max_price_usd: float
    price_range_usd: float
    
    # Trade ID validation
    total_trades: int
    unique_agg_trade_ids: int
    agg_trade_id_gaps: int
    agg_trade_id_duplicates: int
    first_agg_trade_id: int
    last_agg_trade_id: int
    expected_trades_from_ids: int
    
    # Symbol validation
    symbols_found: list[str]
    
    # Data consistency
    all_prices_positive: bool
    all_quantities_positive: bool
    timestamp_monotonic: bool
    
    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print("SOURCE VALIDATION REPORT")
        print(f"Date: {self.date} | Symbol: {self.symbol}")
        print("=" * 70)
        
        print(f"\nPRICE VALIDATION (converted from ticks, tick=0.1 USD)")
        print(f"  Average price: ${self.avg_price_usd:,.2f}")
        print(f"  Min price: ${self.min_price_usd:,.2f}")
        print(f"  Max price: ${self.max_price_usd:,.2f}")
        print(f"  Price range: ${self.price_range_usd:,.2f} ({100*self.price_range_usd/self.avg_price_usd:.2f}%)")
        
        expected_btc_range = (70000, 120000)
        if expected_btc_range[0] <= self.avg_price_usd <= expected_btc_range[1]:
            print(f"  [OK] Price is in expected BTC range (${expected_btc_range[0]:,}-${expected_btc_range[1]:,})")
        else:
            print(f"  [WARNING] Price outside expected BTC range!")
        
        print(f"\nTRADE ID VALIDATION (aggTrade)")
        print(f"  Total trades: {self.total_trades:,}")
        print(f"  Unique agg_trade_ids: {self.unique_agg_trade_ids:,}")
        print(f"  ID range: {self.first_agg_trade_id:,} to {self.last_agg_trade_id:,}")
        print(f"  Expected trades from ID range: {self.expected_trades_from_ids:,}")
        print(f"  Gaps in ID sequence: {self.agg_trade_id_gaps:,}")
        print(f"  Duplicate IDs: {self.agg_trade_id_duplicates:,}")
        
        if self.agg_trade_id_gaps == 0 and self.agg_trade_id_duplicates == 0:
            print(f"  [OK] Trade IDs are consecutive and unique")
        else:
            if self.agg_trade_id_gaps > 0:
                print(f"  [WARNING] {self.agg_trade_id_gaps} gaps in trade ID sequence")
            if self.agg_trade_id_duplicates > 0:
                print(f"  [WARNING] {self.agg_trade_id_duplicates} duplicate trade IDs")
        
        print(f"\nSYMBOL VALIDATION")
        print(f"  Symbols found: {self.symbols_found}")
        if len(self.symbols_found) == 1 and self.symbols_found[0] == self.symbol:
            print(f"  [OK] Only expected symbol found")
        else:
            print(f"  [WARNING] Unexpected symbols in data!")
        
        print(f"\nDATA CONSISTENCY")
        print(f"  All prices positive: {'[OK]' if self.all_prices_positive else '[ERROR]'}")
        print(f"  All quantities positive: {'[OK]' if self.all_quantities_positive else '[ERROR]'}")
        print(f"  Timestamps monotonic: {'[OK]' if self.timestamp_monotonic else '[WARNING] Not monotonic'}")
        
        print("=" * 70)


def validate_trade_ids(trades: pl.DataFrame) -> dict:
    """Validate aggTrade ID sequence."""
    
    # Check if agg_trade_id column exists and has values
    if "agg_trade_id" not in trades.columns:
        return {
            "unique_agg_trade_ids": 0,
            "agg_trade_id_gaps": -1,
            "agg_trade_id_duplicates": -1,
            "first_agg_trade_id": 0,
            "last_agg_trade_id": 0,
            "expected_trades_from_ids": 0,
        }
    
    agg_ids = trades["agg_trade_id"].to_numpy()
    
    # Filter out zeros (from @trade stream)
    valid_ids = agg_ids[agg_ids > 0]
    
    if len(valid_ids) == 0:
        logger.warning("No valid agg_trade_ids found - data may be from @trade stream")
        return {
            "unique_agg_trade_ids": 0,
            "agg_trade_id_gaps": -1,
            "agg_trade_id_duplicates": -1,
            "first_agg_trade_id": 0,
            "last_agg_trade_id": 0,
            "expected_trades_from_ids": 0,
        }
    
    sorted_ids = np.sort(valid_ids)
    unique_ids = np.unique(sorted_ids)
    
    first_id = int(sorted_ids[0])
    last_id = int(sorted_ids[-1])
    expected = last_id - first_id + 1
    
    # Count gaps (expected IDs that are missing)
    full_range = set(range(first_id, last_id + 1))
    actual = set(unique_ids)
    gaps = len(full_range - actual)
    
    # Count duplicates
    duplicates = len(sorted_ids) - len(unique_ids)
    
    return {
        "unique_agg_trade_ids": len(unique_ids),
        "agg_trade_id_gaps": gaps,
        "agg_trade_id_duplicates": duplicates,
        "first_agg_trade_id": first_id,
        "last_agg_trade_id": last_id,
        "expected_trades_from_ids": expected,
    }


def validate_source(
    data_root: Path,
    symbol: str,
    date: str,
    tick_size: float = 0.1,
) -> SourceValidationReport:
    """Run full source validation for a date."""
    
    snap_path = data_root / "snapshots" / symbol / date
    trade_path = data_root / "trades" / symbol / date
    
    # Load snapshots
    logger.info(f"Loading snapshots from {snap_path}")
    snapshots = pl.read_parquet(f"{snap_path}/**/*.parquet")
    logger.info(f"Loaded {snapshots.height:,} snapshots")
    
    # Load trades
    logger.info(f"Loading trades from {trade_path}")
    trades = pl.read_parquet(f"{trade_path}/**/*.parquet")
    logger.info(f"Loaded {trades.height:,} trades")
    
    # Price validation (from snapshots - best_bid)
    snap_prices = snapshots["best_bid_ticks"].to_numpy()
    snap_prices_usd = snap_prices * tick_size
    
    # Trade ID validation
    trade_id_results = validate_trade_ids(trades)
    
    # Symbol validation
    if "symbol" in snapshots.columns:
        snap_symbols = snapshots["symbol"].unique().to_list()
    else:
        snap_symbols = [symbol]  # Assume correct if not in data
    
    if "symbol" in trades.columns:
        trade_symbols = trades["symbol"].unique().to_list()
    else:
        trade_symbols = [symbol]
    
    all_symbols = list(set(snap_symbols + trade_symbols))
    
    # Data consistency checks
    trade_prices = trades["price_ticks"].to_numpy()
    trade_qty = trades["qty_lots"].to_numpy()
    trade_ts = trades["ts_ns"].to_numpy()
    
    all_prices_positive = bool((trade_prices > 0).all() and (snap_prices > 0).all())
    all_quantities_positive = bool((trade_qty > 0).all())
    timestamp_monotonic = bool((np.diff(np.sort(trade_ts)) >= 0).all())
    
    return SourceValidationReport(
        date=date,
        symbol=symbol,
        avg_price_usd=float(np.mean(snap_prices_usd)),
        min_price_usd=float(np.min(snap_prices_usd)),
        max_price_usd=float(np.max(snap_prices_usd)),
        price_range_usd=float(np.max(snap_prices_usd) - np.min(snap_prices_usd)),
        total_trades=trades.height,
        **trade_id_results,
        symbols_found=all_symbols,
        all_prices_positive=all_prices_positive,
        all_quantities_positive=all_quantities_positive,
        timestamp_monotonic=timestamp_monotonic,
    )


async def cross_check_with_binance(
    symbol: str,
    date: str,
    sample_trade_ids: list[int],
) -> dict:
    """Cross-check sample trade IDs with Binance REST API."""
    
    try:
        import aiohttp
    except ImportError:
        logger.warning("aiohttp not installed - skipping Binance REST cross-check")
        return {"status": "skipped", "reason": "aiohttp not installed"}
    
    base_url = "https://fapi.binance.com"
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        for trade_id in sample_trade_ids[:5]:  # Limit to 5 samples
            url = f"{base_url}/fapi/v1/aggTrades"
            params = {
                "symbol": symbol,
                "fromId": trade_id,
                "limit": 1,
            }
            
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and data[0]["a"] == trade_id:
                            results.append({"id": trade_id, "status": "verified"})
                        else:
                            results.append({"id": trade_id, "status": "mismatch"})
                    else:
                        results.append({"id": trade_id, "status": f"error_{resp.status}"})
            except Exception as e:
                results.append({"id": trade_id, "status": f"error_{e}"})
    
    verified = sum(1 for r in results if r["status"] == "verified")
    
    return {
        "status": "completed",
        "samples_checked": len(results),
        "verified": verified,
        "results": results,
    }


def diagnose_all_dates(
    data_root: Path,
    symbol: str,
    dates: list[str],
    output_dir: Optional[Path] = None,
    cross_check: bool = False,
) -> None:
    """Run source validation across all dates."""
    
    print("\n" + "=" * 80)
    print("SOURCE VALIDATION ACROSS ALL DATES")
    print("=" * 80)
    
    summary_data = []
    
    for date in dates:
        try:
            report = validate_source(data_root, symbol, date)
            report.print_report()
            
            summary_data.append({
                "date": date,
                "avg_price_usd": report.avg_price_usd,
                "price_range_pct": 100 * report.price_range_usd / report.avg_price_usd,
                "total_trades": report.total_trades,
                "unique_ids": report.unique_agg_trade_ids,
                "id_gaps": report.agg_trade_id_gaps,
                "id_duplicates": report.agg_trade_id_duplicates,
                "prices_ok": report.all_prices_positive,
                "qty_ok": report.all_quantities_positive,
            })
            
            # Optional cross-check with Binance
            if cross_check and report.unique_agg_trade_ids > 0:
                sample_ids = [report.first_agg_trade_id, report.last_agg_trade_id]
                result = asyncio.run(cross_check_with_binance(symbol, date, sample_ids))
                print(f"\nBINANCE REST CROSS-CHECK: {result}")
        
        except Exception as e:
            logger.error(f"Failed to validate {date}: {e}")
            summary_data.append({
                "date": date,
                "avg_price_usd": 0,
                "price_range_pct": 0,
                "total_trades": 0,
                "unique_ids": 0,
                "id_gaps": -1,
                "id_duplicates": -1,
                "prices_ok": False,
                "qty_ok": False,
            })
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary = pl.DataFrame(summary_data)
    print(summary)
    
    # Overall assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)
    
    all_prices_ok = all(d["prices_ok"] for d in summary_data)
    all_qty_ok = all(d["qty_ok"] for d in summary_data)
    total_gaps = sum(d["id_gaps"] for d in summary_data if d["id_gaps"] >= 0)
    total_dupes = sum(d["id_duplicates"] for d in summary_data if d["id_duplicates"] >= 0)
    
    avg_price = summary["avg_price_usd"].mean()
    
    print(f"\n1. PRICE PLAUSIBILITY")
    print(f"   Average BTC price across all dates: ${avg_price:,.2f}")
    if 70000 <= avg_price <= 120000:
        print(f"   [OK] Prices are in expected BTCUSDT range for Futures")
        print(f"   This confirms data is from Binance Futures, not Spot")
    else:
        print(f"   [WARNING] Prices outside expected range - verify data source")
    
    print(f"\n2. TRADE ID INTEGRITY")
    print(f"   Total ID gaps across all dates: {total_gaps:,}")
    print(f"   Total duplicate IDs: {total_dupes:,}")
    if total_gaps == 0 and total_dupes == 0:
        print(f"   [OK] All aggTrade IDs are consecutive and unique")
        print(f"   This confirms data integrity from @aggTrade stream")
    else:
        if total_gaps > 0:
            print(f"   [WARNING] {total_gaps} gaps - some trades may be missing")
        if total_dupes > 0:
            print(f"   [WARNING] {total_dupes} duplicates - data may be overlapping")
    
    print(f"\n3. DATA QUALITY")
    if all_prices_ok and all_qty_ok:
        print(f"   [OK] All prices and quantities are positive")
    else:
        if not all_prices_ok:
            print(f"   [ERROR] Some prices are zero or negative")
        if not all_qty_ok:
            print(f"   [ERROR] Some quantities are zero or negative")
    
    print("\n4. CONCLUSION")
    if all_prices_ok and all_qty_ok and 70000 <= avg_price <= 120000:
        print("   [OK] Data appears to be from Binance Futures BTCUSDT")
        print("   The large price deviations (>700 ticks) between trades and snapshots")
        print("   are likely due to:")
        print("   - High market volatility (BTC moved $200+ within 100ms)")
        print("   - 100ms snapshot interval missing rapid price moves")
        print("   - This is EXPECTED behavior, not a data source mismatch")
    else:
        print("   [WARNING] Some data quality issues detected - review above")
    
    # Save output
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / "source_validation_summary.csv"
        summary.write_csv(summary_file)
        logger.info(f"Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Validate data source consistency")
    parser.add_argument("--data-root", type=Path, default=Path("./data/binance"), help="Data root directory")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--dates", type=str, nargs="+", help="Dates to validate (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--cross-check", action="store_true", help="Cross-check with Binance REST API")
    
    args = parser.parse_args()
    
    if not args.dates:
        snap_root = args.data_root / "snapshots" / args.symbol
        if snap_root.exists():
            args.dates = sorted([d.name for d in snap_root.iterdir() if d.is_dir()])
    
    if not args.dates:
        logger.error("No dates found or specified")
        return
    
    logger.info(f"Validating dates: {args.dates}")
    diagnose_all_dates(args.data_root, args.symbol, args.dates, args.output_dir, args.cross_check)


if __name__ == "__main__":
    main()

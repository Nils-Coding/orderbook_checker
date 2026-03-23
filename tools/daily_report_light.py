#!/usr/bin/env python3
"""
Memory-efficient daily report generator.

Processes files in small batches to avoid OOM on limited memory VMs.
"""

import argparse
import gc
import json
import re
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DailyReport:
    """Complete daily quality report."""
    
    date: str
    symbol: str
    generated_at: str
    total_snapshots: int
    total_trades: int
    recording_start: str
    recording_end: str
    recording_duration_h: float
    gaps_found: int
    max_gap_ms: float
    total_gap_time_s: float
    snapshot_completeness_pct: float
    trade_id_gaps: int
    trade_completeness_pct: float
    crossed_book_events: int
    min_spread_ticks: int
    max_spread_ticks: int
    avg_spread_ticks: float
    resync_count: int
    resync_reasons: dict
    avg_resync_duration_s: float
    verdict: str
    issues: list
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    def to_text(self) -> str:
        lines = [
            "=" * 65,
            "ORDERBOOK RECORDER - DAILY QUALITY REPORT",
            f"Date: {self.date}",
            f"Symbol: {self.symbol}",
            f"Generated: {self.generated_at}",
            "=" * 65,
            "",
            "RECORDING STATISTICS",
            f"  Total snapshots: {self.total_snapshots:,}",
            f"  Total trades: {self.total_trades:,}",
            f"  Recording period: {self.recording_start} - {self.recording_end}",
            f"  Duration: {self.recording_duration_h:.2f} hours",
            "",
            "GAP ANALYSIS",
            f"  Gaps detected (>150ms): {self.gaps_found}",
            f"  Max gap: {self.max_gap_ms:.1f}ms ({self.max_gap_ms/1000:.2f}s)",
            f"  Total gap time: {self.total_gap_time_s:.2f}s",
            f"  Snapshot completeness: {self.snapshot_completeness_pct:.2f}%",
            "",
            "TRADE ANALYSIS",
            f"  Trade ID sequence gaps: {self.trade_id_gaps}",
            f"  Trade completeness: {self.trade_completeness_pct:.2f}%",
            "",
            "SPREAD SANITY",
            f"  Crossed book events: {self.crossed_book_events}",
            f"  Spread range: {self.min_spread_ticks} - {self.max_spread_ticks} ticks",
            f"  Average spread: {self.avg_spread_ticks:.2f} ticks",
            "",
            "RESYNC ANALYSIS",
            f"  Total resyncs: {self.resync_count}",
            f"  Average resync duration: {self.avg_resync_duration_s:.2f}s",
            "  Reasons:",
        ]
        for reason, count in sorted(self.resync_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"    - {reason}: {count}")
        lines.extend([
            "",
            "=" * 65,
            f"VERDICT: {self.verdict}",
        ])
        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        lines.append("=" * 65)
        return "\n".join(lines)


def process_snapshots_streaming(data_root: Path, symbol: str, date: str) -> dict:
    """Process snapshots in streaming fashion to minimize memory."""
    
    base_path = data_root / "snapshots" / f"symbol={symbol}" / f"date={date}"
    
    if not base_path.exists():
        return {"count": 0, "first_ts": 0, "last_ts": 0, "gaps": 0, "max_gap": 0, 
                "total_gap": 0, "crossed": 0, "spreads": [], "min_spread": 0, 
                "max_spread": 0, "spread_sum": 0}
    
    parquet_files = sorted(base_path.rglob("*.parquet"))
    logger.info(f"Processing {len(parquet_files)} snapshot files...")
    
    total_count = 0
    first_ts = None
    last_ts = None
    prev_ts = None
    gaps = []
    crossed_count = 0
    spread_sum = 0
    min_spread = float('inf')
    max_spread = 0
    
    for i, f in enumerate(parquet_files):
        try:
            # Read only needed columns
            table = pq.read_table(f, columns=["ts_ns", "best_bid_ticks", "best_ask_ticks"])
            
            ts_col = table.column("ts_ns").to_numpy()
            bid_col = table.column("best_bid_ticks").to_numpy()
            ask_col = table.column("best_ask_ticks").to_numpy()
            
            if len(ts_col) == 0:
                continue
            
            total_count += len(ts_col)
            
            if first_ts is None:
                first_ts = ts_col[0]
            last_ts = ts_col[-1]
            
            # Gap detection within file
            deltas = np.diff(ts_col) / 1_000_000  # ms
            file_gaps = deltas[deltas > 150]
            gaps.extend(file_gaps.tolist())
            
            # Gap between files
            if prev_ts is not None:
                inter_file_gap = (ts_col[0] - prev_ts) / 1_000_000
                if inter_file_gap > 150:
                    gaps.append(inter_file_gap)
            prev_ts = ts_col[-1]
            
            # Spread analysis
            spreads = ask_col - bid_col
            crossed_count += int((spreads < 0).sum())
            valid_spreads = spreads[spreads > 0]
            if len(valid_spreads) > 0:
                spread_sum += valid_spreads.sum()
                min_spread = min(min_spread, valid_spreads.min())
                max_spread = max(max_spread, valid_spreads.max())
            
            # Free memory
            del table, ts_col, bid_col, ask_col, deltas, spreads
            gc.collect()
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(parquet_files)} files...")
                
        except Exception as e:
            logger.warning(f"Failed to process {f}: {e}")
    
    return {
        "count": total_count,
        "first_ts": first_ts or 0,
        "last_ts": last_ts or 0,
        "gaps": len(gaps),
        "max_gap": max(gaps) if gaps else 0,
        "total_gap": sum(gaps) / 1000 if gaps else 0,
        "crossed": crossed_count,
        "min_spread": int(min_spread) if min_spread != float('inf') else 0,
        "max_spread": int(max_spread),
        "spread_sum": spread_sum,
    }


def process_trades_streaming(data_root: Path, symbol: str, date: str) -> dict:
    """Process trades in streaming fashion to minimize memory."""
    
    base_path = data_root / "trades" / f"symbol={symbol}" / f"date={date}"
    
    if not base_path.exists():
        return {"count": 0, "id_gaps": 0, "completeness": 0}
    
    parquet_files = sorted(base_path.rglob("*.parquet"))
    logger.info(f"Processing {len(parquet_files)} trade files...")
    
    total_count = 0
    all_ids = []
    has_agg_trade_id = False
    
    for i, f in enumerate(parquet_files):
        try:
            # Check schema first
            pf = pq.ParquetFile(f)
            schema_names = [field.name for field in pf.schema_arrow]
            
            if "agg_trade_id" in schema_names:
                has_agg_trade_id = True
                table = pq.read_table(f, columns=["agg_trade_id"])
                ids = table.column("agg_trade_id").to_numpy()
                # Filter out 0s and NaNs
                valid_ids = ids[(ids > 0) & ~np.isnan(ids)]
                if len(valid_ids) > 0:
                    all_ids.append((valid_ids.min(), valid_ids.max(), len(valid_ids)))
                total_count += len(ids)
                del table, ids
            else:
                table = pq.read_table(f, columns=["ts_ns"])
                total_count += table.num_rows
                del table
            
            gc.collect()
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(parquet_files)} files...")
                
        except Exception as e:
            logger.warning(f"Failed to process {f}: {e}")
    
    if not has_agg_trade_id or not all_ids:
        return {"count": total_count, "id_gaps": -1, "completeness": -1}
    
    # Calculate gaps from min/max ranges
    sorted_ranges = sorted(all_ids, key=lambda x: x[0])
    total_ids = sum(r[2] for r in sorted_ranges)
    overall_min = sorted_ranges[0][0]
    overall_max = sorted_ranges[-1][1]
    expected = overall_max - overall_min + 1
    
    id_gaps = 0
    prev_max = None
    for min_id, max_id, count in sorted_ranges:
        if prev_max is not None and min_id > prev_max + 1:
            id_gaps += 1
        prev_max = max_id
    
    completeness = 100 * total_ids / expected if expected > 0 else 100
    
    return {"count": total_count, "id_gaps": id_gaps, "completeness": completeness}


def analyze_resyncs(log_path: Path, date: str) -> dict:
    """Analyze resync events from log file."""
    
    if not log_path.exists():
        return {"resync_count": 0, "reasons": {}, "avg_duration_s": 0}
    
    resync_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Resync triggered: (.*)")
    live_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*SYNCING -> LIVE")
    
    reasons = {}
    resync_times = []
    live_times = []
    
    try:
        with open(log_path, "r") as f:
            for line in f:
                if date not in line:
                    continue
                match = resync_pattern.search(line)
                if match:
                    ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                    reason = match.group(2).strip()
                    resync_times.append(ts)
                    reasons[reason] = reasons.get(reason, 0) + 1
                match = live_pattern.search(line)
                if match:
                    ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                    live_times.append(ts)
    except Exception as e:
        logger.warning(f"Failed to parse logs: {e}")
    
    avg_duration = 0
    if resync_times and live_times:
        durations = []
        for rt in resync_times:
            following = [lt for lt in live_times if lt > rt]
            if following:
                durations.append((following[0] - rt).total_seconds())
        if durations:
            avg_duration = sum(durations) / len(durations)
    
    return {"resync_count": len(resync_times), "reasons": reasons, "avg_duration_s": avg_duration}


def determine_verdict(data: dict) -> tuple[str, list]:
    """Determine overall verdict and list issues."""
    issues = []
    
    if data.get("crossed", 0) > 0:
        issues.append(f"Crossed book detected {data['crossed']} times")
    if data.get("snapshot_completeness", 100) < 95:
        issues.append(f"Snapshot completeness below 95%: {data['snapshot_completeness']:.1f}%")
    if data.get("trade_completeness", 100) > 0 and data.get("trade_completeness", 100) < 99:
        issues.append(f"Trade completeness below 99%: {data['trade_completeness']:.1f}%")
    if data.get("resync_count", 0) > 50:
        issues.append(f"High resync count: {data['resync_count']}")
    if data.get("max_gap", 0) > 60000:
        issues.append(f"Large gap detected: {data['max_gap']/1000:.1f}s")
    
    if not issues:
        return "HEALTHY", []
    elif len(issues) <= 2:
        return "WARNING", issues
    else:
        return "CRITICAL", issues


def generate_report(data_root: Path, symbol: str, date: str, log_path: Optional[Path] = None) -> DailyReport:
    """Generate a complete daily report with minimal memory footprint."""
    
    logger.info(f"Generating report for {symbol} on {date}")
    
    snap_stats = process_snapshots_streaming(data_root, symbol, date)
    gc.collect()
    
    trade_stats = process_trades_streaming(data_root, symbol, date)
    gc.collect()
    
    if log_path is None:
        log_path = data_root / "logs" / "recorder.log"
    resync_stats = analyze_resyncs(log_path, date)
    
    # Calculate derived metrics
    recording_start = ""
    recording_end = ""
    duration_h = 0
    completeness = 0
    
    if snap_stats["first_ts"] and snap_stats["last_ts"]:
        from datetime import datetime as dt
        start_ts = dt.fromtimestamp(snap_stats["first_ts"] / 1e9)
        end_ts = dt.fromtimestamp(snap_stats["last_ts"] / 1e9)
        recording_start = start_ts.strftime("%H:%M:%S")
        recording_end = end_ts.strftime("%H:%M:%S")
        duration_h = (end_ts - start_ts).total_seconds() / 3600
        
        expected_count = (snap_stats["last_ts"] - snap_stats["first_ts"]) / (100 * 1_000_000)
        completeness = min(100, 100 * snap_stats["count"] / expected_count) if expected_count > 0 else 0
    
    avg_spread = snap_stats["spread_sum"] / snap_stats["count"] if snap_stats["count"] > 0 else 0
    
    verdict_data = {
        "crossed": snap_stats["crossed"],
        "snapshot_completeness": completeness,
        "trade_completeness": trade_stats["completeness"],
        "resync_count": resync_stats["resync_count"],
        "max_gap": snap_stats["max_gap"],
    }
    verdict, issues = determine_verdict(verdict_data)
    
    return DailyReport(
        date=date,
        symbol=symbol,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_snapshots=snap_stats["count"],
        total_trades=trade_stats["count"],
        recording_start=recording_start,
        recording_end=recording_end,
        recording_duration_h=duration_h,
        gaps_found=snap_stats["gaps"],
        max_gap_ms=snap_stats["max_gap"],
        total_gap_time_s=snap_stats["total_gap"],
        snapshot_completeness_pct=completeness,
        trade_id_gaps=trade_stats["id_gaps"],
        trade_completeness_pct=trade_stats["completeness"],
        crossed_book_events=snap_stats["crossed"],
        min_spread_ticks=snap_stats["min_spread"],
        max_spread_ticks=snap_stats["max_spread"],
        avg_spread_ticks=avg_spread,
        resync_count=resync_stats["resync_count"],
        resync_reasons=resync_stats["reasons"],
        avg_resync_duration_s=resync_stats["avg_duration_s"],
        verdict=verdict,
        issues=issues,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate daily quality report (memory-efficient)")
    parser.add_argument("--data-root", type=Path, default=Path("./data"), help="Data root directory")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--date", type=str, help="Date (YYYY-MM-DD), defaults to yesterday")
    parser.add_argument("--output-dir", type=Path, help="Output directory for reports")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text")
    
    args = parser.parse_args()
    
    if args.date is None:
        args.date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    report = generate_report(args.data_root, args.symbol, args.date)
    
    if args.json:
        print(report.to_json())
    else:
        print(report.to_text())
    
    # Save to output directory
    output_dir = args.output_dir or args.data_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_file = output_dir / f"daily_report_{args.date}.txt"
    json_file = output_dir / f"daily_report_{args.date}.json"
    
    with open(txt_file, "w") as f:
        f.write(report.to_text())
    with open(json_file, "w") as f:
        f.write(report.to_json())
    
    logger.info(f"Reports saved to {output_dir}")


if __name__ == "__main__":
    main()

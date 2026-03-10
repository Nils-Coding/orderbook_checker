#!/usr/bin/env python3
"""
Daily quality report generator for orderbook recorder.

Generates a comprehensive report including:
- Uptime and recording statistics
- Gap analysis
- Trade completeness
- Spread sanity checks
- Resync analysis from logs
"""

import argparse
import json
import re
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
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


@dataclass
class DailyReport:
    """Complete daily quality report."""
    
    date: str
    symbol: str
    generated_at: str
    
    # Recording stats
    total_snapshots: int
    total_trades: int
    recording_start: str
    recording_end: str
    recording_duration_h: float
    
    # Gap analysis
    gaps_found: int
    max_gap_ms: float
    total_gap_time_s: float
    snapshot_completeness_pct: float
    
    # Trade analysis
    trade_id_gaps: int
    trade_completeness_pct: float
    
    # Spread analysis
    crossed_book_events: int
    min_spread_ticks: int
    max_spread_ticks: int
    avg_spread_ticks: float
    
    # Resync analysis (from logs)
    resync_count: int
    resync_reasons: dict
    avg_resync_duration_s: float
    
    # Verdict
    verdict: str
    issues: list
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(asdict(self), indent=2)
    
    def to_text(self) -> str:
        """Convert report to formatted text."""
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


def load_data_for_date(data_root: Path, symbol: str, date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load snapshots and trades for a specific date."""
    
    snapshots_path = data_root / "snapshots" / f"symbol={symbol}" / f"date={date}"
    trades_path = data_root / "trades" / f"symbol={symbol}" / f"date={date}"
    
    snapshots = pd.DataFrame()
    trades = pd.DataFrame()
    
    if snapshots_path.exists():
        parquet_files = list(snapshots_path.rglob("*.parquet"))
        if parquet_files:
            logger.info(f"Loading {len(parquet_files)} snapshot files...")
            dfs = []
            for f in sorted(parquet_files):
                try:
                    df = pq.ParquetFile(f).read().to_pandas()
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")
            if dfs:
                snapshots = pd.concat(dfs, ignore_index=True).sort_values("ts_ns")
                logger.info(f"Loaded {len(snapshots):,} snapshots")
    
    if trades_path.exists():
        parquet_files = list(trades_path.rglob("*.parquet"))
        if parquet_files:
            logger.info(f"Loading {len(parquet_files)} trade files...")
            dfs = []
            for f in sorted(parquet_files):
                try:
                    df = pq.ParquetFile(f).read().to_pandas()
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")
            if dfs:
                trades = pd.concat(dfs, ignore_index=True).sort_values("ts_ns")
                logger.info(f"Loaded {len(trades):,} trades")
    
    return snapshots, trades


def analyze_gaps(snapshots: pd.DataFrame, expected_interval_ms: int = 100, tolerance_ms: int = 50) -> dict:
    """Analyze gaps in snapshot timestamps."""
    
    if len(snapshots) < 2:
        return {"gaps_found": 0, "max_gap_ms": 0, "total_gap_time_s": 0, "completeness_pct": 0}
    
    deltas = np.diff(snapshots["ts_ns"].values) / 1_000_000
    threshold = expected_interval_ms + tolerance_ms
    gaps = deltas[deltas > threshold]
    
    expected_count = (snapshots["ts_ns"].max() - snapshots["ts_ns"].min()) / (expected_interval_ms * 1_000_000)
    completeness = min(100, 100 * len(snapshots) / expected_count) if expected_count > 0 else 0
    
    return {
        "gaps_found": len(gaps),
        "max_gap_ms": float(gaps.max()) if len(gaps) > 0 else 0,
        "total_gap_time_s": float(gaps.sum()) / 1000 if len(gaps) > 0 else 0,
        "completeness_pct": completeness,
    }


def analyze_trades(trades: pd.DataFrame) -> dict:
    """Analyze trade ID sequence."""
    
    if len(trades) == 0:
        return {"trade_id_gaps": 0, "completeness_pct": 0}
    
    if "agg_trade_id" not in trades.columns or trades["agg_trade_id"].isna().all() or (trades["agg_trade_id"] == 0).all():
        return {"trade_id_gaps": -1, "completeness_pct": -1}  # -1 means not available
    
    ids = trades["agg_trade_id"].dropna().sort_values().values
    if len(ids) < 2:
        return {"trade_id_gaps": 0, "completeness_pct": 100}
    
    diffs = np.diff(ids)
    gap_count = int((diffs > 1).sum())
    
    expected = int(ids[-1] - ids[0] + 1)
    completeness = 100 * len(ids) / expected if expected > 0 else 100
    
    return {
        "trade_id_gaps": gap_count,
        "completeness_pct": completeness,
    }


def analyze_spreads(snapshots: pd.DataFrame) -> dict:
    """Analyze spread sanity."""
    
    if len(snapshots) == 0:
        return {"crossed_book_events": 0, "min_spread": 0, "max_spread": 0, "avg_spread": 0}
    
    spreads = snapshots["best_ask_ticks"] - snapshots["best_bid_ticks"]
    crossed = int((spreads < 0).sum())
    valid = spreads[spreads > 0]
    
    return {
        "crossed_book_events": crossed,
        "min_spread": int(valid.min()) if len(valid) > 0 else 0,
        "max_spread": int(valid.max()) if len(valid) > 0 else 0,
        "avg_spread": float(valid.mean()) if len(valid) > 0 else 0,
    }


def analyze_resyncs(log_path: Path, date: str) -> dict:
    """Analyze resync events from log file."""
    
    resyncs = []
    resync_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Resync triggered: (.*)")
    live_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*SYNCING -> LIVE")
    
    if not log_path.exists():
        return {"resync_count": 0, "reasons": {}, "avg_duration_s": 0}
    
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
    
    return {
        "resync_count": len(resync_times),
        "reasons": reasons,
        "avg_duration_s": avg_duration,
    }


def determine_verdict(report_data: dict) -> tuple[str, list]:
    """Determine overall verdict and list issues."""
    
    issues = []
    
    if report_data["crossed_book_events"] > 0:
        issues.append(f"Crossed book detected {report_data['crossed_book_events']} times")
    
    if report_data["snapshot_completeness_pct"] < 95:
        issues.append(f"Snapshot completeness below 95%: {report_data['snapshot_completeness_pct']:.1f}%")
    
    if report_data.get("trade_completeness_pct", 100) > 0 and report_data.get("trade_completeness_pct", 100) < 99:
        issues.append(f"Trade completeness below 99%: {report_data['trade_completeness_pct']:.1f}%")
    
    if report_data["resync_count"] > 50:
        issues.append(f"High resync count: {report_data['resync_count']}")
    
    if report_data["max_gap_ms"] > 60000:  # > 1 minute
        issues.append(f"Large gap detected: {report_data['max_gap_ms']/1000:.1f}s")
    
    if not issues:
        return "HEALTHY", []
    elif len(issues) <= 2:
        return "WARNING", issues
    else:
        return "CRITICAL", issues


def generate_report(
    data_root: Path,
    symbol: str,
    date: str,
    log_path: Optional[Path] = None,
) -> DailyReport:
    """Generate a complete daily report."""
    
    logger.info(f"Generating report for {symbol} on {date}")
    
    snapshots, trades = load_data_for_date(data_root, symbol, date)
    
    gap_analysis = analyze_gaps(snapshots)
    trade_analysis = analyze_trades(trades)
    spread_analysis = analyze_spreads(snapshots)
    
    if log_path is None:
        log_path = data_root / "logs" / "recorder.log"
    resync_analysis = analyze_resyncs(log_path, date)
    
    recording_start = ""
    recording_end = ""
    duration_h = 0
    
    if len(snapshots) > 0:
        start_ts = pd.Timestamp(snapshots["ts_ns"].min(), unit="ns")
        end_ts = pd.Timestamp(snapshots["ts_ns"].max(), unit="ns")
        recording_start = start_ts.strftime("%H:%M:%S")
        recording_end = end_ts.strftime("%H:%M:%S")
        duration_h = (end_ts - start_ts).total_seconds() / 3600
    
    report_data = {
        "crossed_book_events": spread_analysis["crossed_book_events"],
        "snapshot_completeness_pct": gap_analysis["completeness_pct"],
        "trade_completeness_pct": trade_analysis["completeness_pct"],
        "resync_count": resync_analysis["resync_count"],
        "max_gap_ms": gap_analysis["max_gap_ms"],
    }
    
    verdict, issues = determine_verdict(report_data)
    
    return DailyReport(
        date=date,
        symbol=symbol,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_snapshots=len(snapshots),
        total_trades=len(trades),
        recording_start=recording_start,
        recording_end=recording_end,
        recording_duration_h=duration_h,
        gaps_found=gap_analysis["gaps_found"],
        max_gap_ms=gap_analysis["max_gap_ms"],
        total_gap_time_s=gap_analysis["total_gap_time_s"],
        snapshot_completeness_pct=gap_analysis["completeness_pct"],
        trade_id_gaps=trade_analysis["trade_id_gaps"],
        trade_completeness_pct=trade_analysis["completeness_pct"],
        crossed_book_events=spread_analysis["crossed_book_events"],
        min_spread_ticks=spread_analysis["min_spread"],
        max_spread_ticks=spread_analysis["max_spread"],
        avg_spread_ticks=spread_analysis["avg_spread"],
        resync_count=resync_analysis["resync_count"],
        resync_reasons=resync_analysis["reasons"],
        avg_resync_duration_s=resync_analysis["avg_duration_s"],
        verdict=verdict,
        issues=issues,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate daily quality report")
    parser.add_argument("--data-root", type=Path, default=Path("./data"), help="Data root directory")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--date", type=str, help="Date (YYYY-MM-DD), defaults to yesterday")
    parser.add_argument("--output", type=Path, help="Output file path (optional)")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text")
    
    args = parser.parse_args()
    
    if args.date is None:
        args.date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    report = generate_report(args.data_root, args.symbol, args.date)
    
    if args.json:
        output = report.to_json()
    else:
        output = report.to_text()
    
    print(output)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        logger.info(f"Report saved to {args.output}")
    
    default_output = args.data_root / "reports" / f"daily_report_{args.date}.txt"
    default_output.parent.mkdir(parents=True, exist_ok=True)
    with open(default_output, "w") as f:
        f.write(report.to_text())
    
    json_output = args.data_root / "reports" / f"daily_report_{args.date}.json"
    with open(json_output, "w") as f:
        f.write(report.to_json())
    
    logger.info(f"Reports saved to {default_output.parent}")


if __name__ == "__main__":
    main()

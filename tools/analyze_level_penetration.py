#!/usr/bin/env python3
"""
Level-Penetration Analysis: How deep do trades consume orderbook levels?

For each trade, finds the nearest prior snapshot and walks through the
relevant side (bids for sells, asks for buys) to determine how many
levels the trade quantity would consume.

Usage:
    python tools/analyze_level_penetration.py --date 2026-03-25
    python tools/analyze_level_penetration.py --date 2026-03-25 --html report.html
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

QTY_SCALE = 1000  # 1 lot = 0.001 BTC


def load_parquet_files(base_path: Path) -> pd.DataFrame:
    """Load all parquet files under base_path into a single DataFrame."""
    files = sorted(base_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {base_path}")

    dfs = []
    for f in files:
        try:
            tbl = pq.ParquetFile(f).read()
            dfs.append(tbl.to_pandas())
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values("ts_ns", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def calculate_penetration(trade_qty: int, level_qtys: np.ndarray) -> int:
    """Return the 1-based level at which cumulative qty >= trade_qty."""
    cum = 0
    for i, q in enumerate(level_qtys):
        cum += q
        if cum >= trade_qty:
            return i + 1
    return len(level_qtys) + 1  # overflow


def run_analysis(
    data_root: Path,
    symbol: str,
    date: str,
) -> pd.DataFrame:
    """Match trades to snapshots and compute level penetration."""

    snap_path = data_root / "snapshots" / symbol / date
    trade_path = data_root / "trades" / symbol / date

    logger.info(f"Loading snapshots from {snap_path}")
    snapshots = load_parquet_files(snap_path)
    logger.info(f"  {len(snapshots):,} snapshots loaded")

    logger.info(f"Loading trades from {trade_path}")
    trades = load_parquet_files(trade_path)
    logger.info(f"  {len(trades):,} trades loaded")

    logger.info("Matching trades to nearest prior snapshot (merge_asof)...")
    merged = pd.merge_asof(
        trades,
        snapshots[["ts_ns", "bids_qty_lots", "asks_qty_lots"]],
        on="ts_ns",
        direction="backward",
        suffixes=("", "_snap"),
    )

    results = []
    skipped = 0
    total = len(merged)

    for idx, row in merged.iterrows():
        if idx % 200_000 == 0 and idx > 0:
            logger.info(f"  {idx:,}/{total:,} trades processed")

        bids = row.get("bids_qty_lots")
        if bids is None or (pd.api.types.is_scalar(bids) and pd.isna(bids)):
            skipped += 1
            continue

        is_sell = row["is_buyer_maker"]  # True -> taker sells -> consumes bids
        levels = np.asarray(row["bids_qty_lots"] if is_sell else row["asks_qty_lots"])
        pen = calculate_penetration(row["qty_lots"], levels)

        results.append({
            "ts_ns": row["ts_ns"],
            "price_ticks": row["price_ticks"],
            "qty_lots": row["qty_lots"],
            "btc": row["qty_lots"] / QTY_SCALE,
            "side": "bid" if is_sell else "ask",
            "penetration_level": pen,
        })

    if skipped:
        logger.warning(f"Skipped {skipped:,} trades (no matching snapshot)")

    df = pd.DataFrame(results)
    logger.info(f"Penetration computed for {len(df):,} trades")
    return df


def build_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-level (or grouped) summary table.

    If there are <=30 distinct active levels, report individually.
    Otherwise, group into buckets.
    """
    max_level = int(df["penetration_level"].max())
    active_levels = df["penetration_level"].nunique()

    if active_levels <= 30:
        groups = {lvl: (lvl, lvl) for lvl in sorted(df["penetration_level"].unique())}
    else:
        boundaries = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000, max_level + 1]
        groups = {}
        for i in range(len(boundaries) - 1):
            lo, hi = boundaries[i], boundaries[i + 1] - 1
            if lo > max_level:
                break
            groups[f"{lo}-{hi}" if lo != hi else str(lo)] = (lo, hi)

    total_trades = len(df)
    total_btc = df["btc"].sum()

    rows = []
    cum_trades = 0
    cum_btc = 0.0

    for label, (lo, hi) in groups.items():
        mask = (df["penetration_level"] >= lo) & (df["penetration_level"] <= hi)
        subset = df[mask]
        n = len(subset)
        btc = subset["btc"].sum()
        cum_trades += n
        cum_btc += btc

        bid_n = len(subset[subset["side"] == "bid"])
        ask_n = len(subset[subset["side"] == "ask"])
        bid_btc = subset.loc[subset["side"] == "bid", "btc"].sum()
        ask_btc = subset.loc[subset["side"] == "ask", "btc"].sum()

        rows.append({
            "level": label,
            "trades": n,
            "trades_pct": 100 * n / total_trades if total_trades else 0,
            "cum_trades_pct": 100 * cum_trades / total_trades if total_trades else 0,
            "btc": round(btc, 3),
            "btc_pct": 100 * btc / total_btc if total_btc else 0,
            "cum_btc_pct": 100 * cum_btc / total_btc if total_btc else 0,
            "bid_trades": bid_n,
            "ask_trades": ask_n,
            "bid_btc": round(bid_btc, 3),
            "ask_btc": round(ask_btc, 3),
        })

    return pd.DataFrame(rows)


def print_table(table: pd.DataFrame, date: str, total_trades: int, total_btc: float) -> None:
    """Pretty-print the level penetration table."""
    print()
    print("=" * 100)
    print(f"LEVEL PENETRATION ANALYSIS  |  {date}  |  {total_trades:,} trades  |  {total_btc:,.3f} BTC")
    print("=" * 100)
    print(
        f"{'Level':>8} | {'Trades':>8} {'%':>7} {'Cum%':>7} | "
        f"{'BTC':>10} {'%':>7} {'Cum%':>7} | "
        f"{'Bid Tr':>7} {'Ask Tr':>7} | {'Bid BTC':>9} {'Ask BTC':>9}"
    )
    print("-" * 100)

    for _, r in table.iterrows():
        print(
            f"{r['level']:>8} | "
            f"{r['trades']:>8,} {r['trades_pct']:>6.2f}% {r['cum_trades_pct']:>6.2f}% | "
            f"{r['btc']:>10.3f} {r['btc_pct']:>6.2f}% {r['cum_btc_pct']:>6.2f}% | "
            f"{r['bid_trades']:>7,} {r['ask_trades']:>7,} | "
            f"{r['bid_btc']:>9.3f} {r['ask_btc']:>9.3f}"
        )
    print("=" * 100)


def generate_html(table: pd.DataFrame, date: str, total_trades: int, total_btc: float) -> str:
    """Generate a self-contained HTML page with Chart.js visualisation."""

    labels = table["level"].astype(str).tolist()
    bid_trades = table["bid_trades"].tolist()
    ask_trades = table["ask_trades"].tolist()
    bid_btc = table["bid_btc"].tolist()
    ask_btc = table["ask_btc"].tolist()
    cum_pct = table["cum_trades_pct"].tolist()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Level Penetration &mdash; {date}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117; color: #c9d1d9; padding: 24px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; color: #58a6ff; }}
  .subtitle {{ font-size: 0.85rem; color: #8b949e; margin-bottom: 20px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  .chart-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
  .chart-box h2 {{ font-size: 0.9rem; margin-bottom: 10px; color: #c9d1d9; }}
  canvas {{ width: 100% !important; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  th {{ background: #21262d; padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 600; border-bottom: 1px solid #30363d; }}
  td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #1c2128; }}
  .col-level {{ text-align: left; font-weight: 600; color: #58a6ff; }}
  th.col-level {{ text-align: left; }}
  .bid {{ color: #f85149; }}
  .ask {{ color: #3fb950; }}
</style>
</head>
<body>

<h1>Orderbook Level Penetration</h1>
<p class="subtitle">{date} &nbsp;|&nbsp; {total_trades:,} trades &nbsp;|&nbsp; {total_btc:,.3f} BTC total volume</p>

<div class="charts">
  <div class="chart-box">
    <h2>Trade Count by Level &mdash; log scale (Bid / Ask)</h2>
    <canvas id="tradeChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>BTC Volume by Level &mdash; log scale (Bid / Ask)</h2>
    <canvas id="btcChart"></canvas>
  </div>
</div>

<table>
  <thead>
    <tr>
      <th class="col-level">Level</th>
      <th>Trades</th><th>%</th><th>Cum %</th>
      <th>BTC</th><th>%</th><th>Cum %</th>
      <th class="bid">Bid Tr</th><th class="ask">Ask Tr</th>
      <th class="bid">Bid BTC</th><th class="ask">Ask BTC</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>

<script>
const labels = {json.dumps(labels)};
const bidTrades = {json.dumps(bid_trades)};
const askTrades = {json.dumps(ask_trades)};
const bidBtc = {json.dumps(bid_btc)};
const askBtc = {json.dumps(ask_btc)};
const cumPct = {json.dumps(cum_pct)};

const tableData = {table.to_json(orient='records')};

new Chart(document.getElementById('tradeChart'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: [
      {{ label: 'Bid (sells)', data: bidTrades, backgroundColor: 'rgba(248,81,73,0.8)' }},
      {{ label: 'Ask (buys)', data: askTrades, backgroundColor: 'rgba(63,185,80,0.8)' }},
      {{ label: 'Cumulative %', data: cumPct, type: 'line', borderColor: '#58a6ff', borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#58a6ff', yAxisID: 'yCum', tension: 0.3 }},
    ]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
      y: {{ type: 'logarithmic', position: 'left', grid: {{ color: '#21262d' }},
            ticks: {{ color: '#8b949e' }},
            title: {{ display: true, text: 'Trade Count (log)', color: '#8b949e' }} }},
      yCum: {{ position: 'right', min: 90, max: 100,
               ticks: {{ color: '#58a6ff', callback: function(v) {{ return v + '%'; }} }},
               grid: {{ drawOnChartArea: false }} }},
    }},
    plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }}
  }}
}});

new Chart(document.getElementById('btcChart'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: [
      {{ label: 'Bid BTC', data: bidBtc, backgroundColor: 'rgba(248,81,73,0.8)' }},
      {{ label: 'Ask BTC', data: askBtc, backgroundColor: 'rgba(63,185,80,0.8)' }},
      {{ label: 'Cumulative %', data: cumPct, type: 'line', borderColor: '#58a6ff', borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#58a6ff', yAxisID: 'yCum', tension: 0.3 }},
    ]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
      y: {{ type: 'logarithmic', position: 'left', grid: {{ color: '#21262d' }},
            ticks: {{ color: '#8b949e' }},
            title: {{ display: true, text: 'BTC (log)', color: '#8b949e' }} }},
      yCum: {{ position: 'right', min: 70, max: 100,
               ticks: {{ color: '#58a6ff', callback: function(v) {{ return v + '%'; }} }},
               grid: {{ drawOnChartArea: false }} }},
    }},
    plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }}
  }}
}});

const tbody = document.getElementById('tbody');
tableData.forEach(r => {{
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td class="col-level">${{r.level}}</td>
    <td>${{r.trades.toLocaleString()}}</td>
    <td>${{r.trades_pct.toFixed(2)}}%</td>
    <td>${{r.cum_trades_pct.toFixed(2)}}%</td>
    <td>${{r.btc.toFixed(3)}}</td>
    <td>${{r.btc_pct.toFixed(2)}}%</td>
    <td>${{r.cum_btc_pct.toFixed(2)}}%</td>
    <td class="bid">${{r.bid_trades.toLocaleString()}}</td>
    <td class="ask">${{r.ask_trades.toLocaleString()}}</td>
    <td class="bid">${{r.bid_btc.toFixed(3)}}</td>
    <td class="ask">${{r.ask_btc.toFixed(3)}}</td>
  `;
  tbody.appendChild(tr);
}});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Level penetration analysis")
    parser.add_argument("--data-root", type=Path, default=Path("./data/binance"))
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--html", type=Path, default=None, help="Output HTML file path")
    parser.add_argument("--csv", type=Path, default=None, help="Output CSV file path")

    args = parser.parse_args()

    df = run_analysis(args.data_root, args.symbol, args.date)
    if df.empty:
        logger.error("No results")
        sys.exit(1)

    table = build_level_table(df)
    total_trades = len(df)
    total_btc = df["btc"].sum()

    print_table(table, args.date, total_trades, total_btc)

    # Percentiles
    p50 = df["penetration_level"].quantile(0.5)
    p90 = df["penetration_level"].quantile(0.9)
    p99 = df["penetration_level"].quantile(0.99)
    p_max = df["penetration_level"].max()
    print(f"\nPERCENTILES:  P50={p50:.0f}  P90={p90:.0f}  P99={p99:.0f}  Max={p_max}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        logger.info(f"Saved CSV to {args.csv}")

    html_path = args.html or Path(f"reports/level_penetration_{args.date}.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(generate_html(table, args.date, total_trades, total_btc))
    logger.info(f"Saved HTML to {html_path}")
    print(f"\nOpen in browser:  file://{html_path.resolve()}")


if __name__ == "__main__":
    main()

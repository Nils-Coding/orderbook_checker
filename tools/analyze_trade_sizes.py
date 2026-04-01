#!/usr/bin/env python3
"""
Trade Size Distribution Analysis.

Analyses trade sizes by BTC volume, split by bid/ask side,
with hourly heatmap and penetration-level correlation.

Usage:
    python tools/analyze_trade_sizes.py --date 2026-03-25
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

QTY_SCALE = 1000  # 1 lot = 0.001 BTC

SIZE_BUCKETS = [
    ("dust",     0,      0.001),
    ("micro",    0.001,  0.01),
    ("small",    0.01,   0.1),
    ("medium",   0.1,    1.0),
    ("large",    1.0,    10.0),
    ("whale",    10.0,   100.0),
    ("mega",     100.0,  float("inf")),
]


def load_trades(data_root: Path, symbol: str, date: str) -> pd.DataFrame:
    base = data_root / "trades" / symbol / date
    files = sorted(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {base}")
    dfs = [pq.ParquetFile(f).read().to_pandas() for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values("ts_ns").reset_index(drop=True)
    df["btc"] = df["qty_lots"] / QTY_SCALE
    df["side"] = np.where(df["is_buyer_maker"], "bid", "ask")
    df["hour"] = pd.to_datetime(df["ts_ns"], unit="ns", utc=True).dt.hour
    logger.info(f"Loaded {len(df):,} trades ({df['btc'].sum():,.1f} BTC)")
    return df


def assign_bucket(btc: pd.Series) -> pd.Series:
    result = pd.Series("unknown", index=btc.index)
    for name, lo, hi in SIZE_BUCKETS:
        mask = (btc >= lo) & (btc < hi)
        result[mask] = name
    return result


def build_size_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = assign_bucket(df["btc"])

    total_n = len(df)
    total_btc = df["btc"].sum()

    rows = []
    cum_n, cum_btc = 0, 0.0
    for name, lo, hi in SIZE_BUCKETS:
        sub = df[df["bucket"] == name]
        n = len(sub)
        btc = sub["btc"].sum()
        cum_n += n
        cum_btc += btc

        bid = sub[sub["side"] == "bid"]
        ask = sub[sub["side"] == "ask"]

        hi_label = f"{hi:.3f}" if hi < float("inf") else "+"
        rows.append({
            "bucket": name,
            "range": f"{lo:.3f}-{hi_label}",
            "trades": n,
            "trades_pct": 100 * n / total_n if total_n else 0,
            "cum_trades_pct": 100 * cum_n / total_n if total_n else 0,
            "btc": round(btc, 3),
            "btc_pct": 100 * btc / total_btc if total_btc else 0,
            "cum_btc_pct": 100 * cum_btc / total_btc if total_btc else 0,
            "avg_btc": round(btc / n, 4) if n else 0,
            "max_btc": round(sub["btc"].max(), 4) if n else 0,
            "bid_trades": len(bid),
            "ask_trades": len(ask),
            "bid_btc": round(bid["btc"].sum(), 3),
            "ask_btc": round(ask["btc"].sum(), 3),
        })

    return pd.DataFrame(rows)


def build_hourly_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows=bucket, cols=hour, values=trade count."""
    df = df.copy()
    df["bucket"] = assign_bucket(df["btc"])
    bucket_order = [name for name, _, _ in SIZE_BUCKETS]
    piv = df.pivot_table(index="bucket", columns="hour", values="ts_ns", aggfunc="count", fill_value=0)
    piv = piv.reindex(bucket_order).fillna(0).astype(int)
    return piv


def build_hourly_btc_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = assign_bucket(df["btc"])
    bucket_order = [name for name, _, _ in SIZE_BUCKETS]
    piv = df.pivot_table(index="bucket", columns="hour", values="btc", aggfunc="sum", fill_value=0)
    piv = piv.reindex(bucket_order).fillna(0)
    return piv.round(1)


def print_table(table: pd.DataFrame, date: str, total_trades: int, total_btc: float) -> None:
    print()
    print("=" * 120)
    print(f"TRADE SIZE DISTRIBUTION  |  {date}  |  {total_trades:,} trades  |  {total_btc:,.3f} BTC")
    print("=" * 120)
    print(
        f"{'Bucket':>8} {'Range BTC':>16} | {'Trades':>9} {'%':>7} {'Cum%':>7} | "
        f"{'BTC':>12} {'%':>7} {'Cum%':>7} | "
        f"{'Avg BTC':>9} {'Max BTC':>10} | "
        f"{'Bid Tr':>8} {'Ask Tr':>8} | {'Bid BTC':>10} {'Ask BTC':>10}"
    )
    print("-" * 120)
    for _, r in table.iterrows():
        print(
            f"{r['bucket']:>8} {r['range']:>16} | "
            f"{r['trades']:>9,} {r['trades_pct']:>6.2f}% {r['cum_trades_pct']:>6.2f}% | "
            f"{r['btc']:>12,.3f} {r['btc_pct']:>6.2f}% {r['cum_btc_pct']:>6.2f}% | "
            f"{r['avg_btc']:>9.4f} {r['max_btc']:>10.4f} | "
            f"{r['bid_trades']:>8,} {r['ask_trades']:>8,} | "
            f"{r['bid_btc']:>10,.3f} {r['ask_btc']:>10,.3f}"
        )
    print("=" * 120)


def generate_html(
    table: pd.DataFrame,
    heatmap_counts: pd.DataFrame,
    heatmap_btc: pd.DataFrame,
    date: str,
    total_trades: int,
    total_btc: float,
    percentiles: dict,
) -> str:
    chart_table = table[table["trades"] > 0]
    labels = chart_table["bucket"].tolist()
    bid_trades = chart_table["bid_trades"].tolist()
    ask_trades = chart_table["ask_trades"].tolist()
    bid_btc = chart_table["bid_btc"].tolist()
    ask_btc = chart_table["ask_btc"].tolist()
    cum_btc_pct = chart_table["cum_btc_pct"].tolist()

    hours = list(range(24))
    hm_buckets = heatmap_counts.index.tolist()
    hm_data = []
    for hi, h in enumerate(hours):
        for bi, b in enumerate(hm_buckets):
            v = int(heatmap_counts.loc[b, h]) if h in heatmap_counts.columns else 0
            if v > 0:
                hm_data.append({"x": hi, "y": bi, "v": v})

    hm_btc_data = []
    for hi, h in enumerate(hours):
        for bi, b in enumerate(hm_buckets):
            v = float(heatmap_btc.loc[b, h]) if h in heatmap_btc.columns else 0
            if v > 0:
                hm_btc_data.append({"x": hi, "y": bi, "v": round(v, 1)})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trade Size Distribution &mdash; {date}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'SF Mono','Fira Code',monospace; background:#0d1117; color:#c9d1d9; padding:24px; }}
  h1 {{ font-size:1.4rem; margin-bottom:4px; color:#58a6ff; }}
  .sub {{ font-size:0.85rem; color:#8b949e; margin-bottom:20px; }}
  .row {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px; }}
  .box {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }}
  .box h2 {{ font-size:0.9rem; margin-bottom:10px; color:#c9d1d9; }}
  .full {{ margin-bottom:20px; }}
  canvas {{ width:100%!important; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.75rem; background:#161b22; border:1px solid #30363d; border-radius:8px; overflow:hidden; }}
  th {{ background:#21262d; padding:8px 10px; text-align:right; color:#8b949e; font-weight:600; border-bottom:1px solid #30363d; }}
  td {{ padding:6px 10px; text-align:right; border-bottom:1px solid #21262d; }}
  tr:hover td {{ background:#1c2128; }}
  .cl {{ text-align:left; font-weight:600; color:#58a6ff; }}
  th.cl {{ text-align:left; }}
  .bid {{ color:#f85149; }}
  .ask {{ color:#3fb950; }}
  .pbox {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:20px; }}
  .pcard {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; text-align:center; }}
  .pcard .label {{ font-size:0.7rem; color:#8b949e; margin-bottom:4px; }}
  .pcard .val {{ font-size:1.1rem; color:#e6edf3; font-weight:700; }}
</style>
</head>
<body>

<h1>Trade Size Distribution</h1>
<p class="sub">{date} &nbsp;|&nbsp; {total_trades:,} trades &nbsp;|&nbsp; {total_btc:,.3f} BTC</p>

<div class="pbox">
  <div class="pcard"><div class="label">Median</div><div class="val">{percentiles['p50']:.4f} BTC</div></div>
  <div class="pcard"><div class="label">P90</div><div class="val">{percentiles['p90']:.4f} BTC</div></div>
  <div class="pcard"><div class="label">P99</div><div class="val">{percentiles['p99']:.4f} BTC</div></div>
  <div class="pcard"><div class="label">P99.9</div><div class="val">{percentiles['p999']:.3f} BTC</div></div>
  <div class="pcard"><div class="label">Max</div><div class="val">{percentiles['max']:.2f} BTC</div></div>
</div>

<div class="row">
  <div class="box">
    <h2>Trade Count by Size (Bid / Ask) &mdash; log scale</h2>
    <canvas id="countChart"></canvas>
  </div>
  <div class="box">
    <h2>BTC Volume by Size (Bid / Ask) + Cumulative %</h2>
    <canvas id="btcChart"></canvas>
  </div>
</div>

<div class="box full">
  <h2>Hourly BTC Volume by Size Bucket (stacked)</h2>
  <canvas id="hourlyChart" height="100"></canvas>
</div>

<table>
  <thead><tr>
    <th class="cl">Bucket</th><th class="cl">Range BTC</th>
    <th>Trades</th><th>%</th><th>Cum %</th>
    <th>BTC</th><th>%</th><th>Cum %</th>
    <th>Avg BTC</th><th>Max BTC</th>
    <th class="bid">Bid Tr</th><th class="ask">Ask Tr</th>
    <th class="bid">Bid BTC</th><th class="ask">Ask BTC</th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>

<script>
const labels = {json.dumps(labels)};
const bidTr = {json.dumps(bid_trades)};
const askTr = {json.dumps(ask_trades)};
const bidBtc = {json.dumps(bid_btc)};
const askBtc = {json.dumps(ask_btc)};
const cumBtcPct = {json.dumps(cum_btc_pct)};
const tableData = {table.to_json(orient='records')};
const hourLabels = {json.dumps([f"{h:02d}h" for h in range(24)])};
const hourlyBtcByBucket = {json.dumps({name: heatmap_btc.loc[name].tolist() if name in heatmap_btc.index else [0]*24 for name, _, _ in SIZE_BUCKETS if name != "dust"})};

new Chart(document.getElementById('countChart'), {{
  type:'bar',
  data:{{
    labels,
    datasets:[
      {{label:'Bid (sells)',data:bidTr,backgroundColor:'rgba(248,81,73,0.8)'}},
      {{label:'Ask (buys)',data:askTr,backgroundColor:'rgba(63,185,80,0.8)'}},
    ]
  }},
  options:{{
    responsive:true,
    interaction:{{mode:'index',intersect:false}},
    scales:{{
      x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},
      y:{{type:'logarithmic',position:'left',grid:{{color:'#21262d'}},ticks:{{color:'#8b949e'}},
          title:{{display:true,text:'Count (log)',color:'#8b949e'}}}},
    }},
    plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}}
  }}
}});

new Chart(document.getElementById('btcChart'), {{
  type:'bar',
  data:{{
    labels,
    datasets:[
      {{label:'Bid BTC',data:bidBtc,backgroundColor:'rgba(248,81,73,0.8)'}},
      {{label:'Ask BTC',data:askBtc,backgroundColor:'rgba(63,185,80,0.8)'}},
      {{label:'Cumulative BTC %',data:cumBtcPct,type:'line',borderColor:'#58a6ff',borderWidth:2,pointRadius:4,pointBackgroundColor:'#58a6ff',yAxisID:'yCum',tension:0.3}},
    ]
  }},
  options:{{
    responsive:true,
    interaction:{{mode:'index',intersect:false}},
    scales:{{
      x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},
      y:{{position:'left',grid:{{color:'#21262d'}},ticks:{{color:'#8b949e'}},
          title:{{display:true,text:'BTC',color:'#8b949e'}}}},
      yCum:{{position:'right',min:0,max:100,ticks:{{color:'#58a6ff',callback:function(v){{return v+'%';}}}},grid:{{drawOnChartArea:false}}}},
    }},
    plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}}
  }}
}});

const bucketColors = {{micro:'#484f58',small:'#388bfd',medium:'#d29922',large:'#f85149',whale:'#a371f7',mega:'#ff7b72'}};
const hourlyDatasets = Object.entries(hourlyBtcByBucket).map(([name,data])=>({{
  label:name, data, backgroundColor:bucketColors[name]||'#8b949e', stack:'s'
}}));
new Chart(document.getElementById('hourlyChart'), {{
  type:'bar',
  data:{{ labels:hourLabels, datasets:hourlyDatasets }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    interaction:{{ mode:'index', intersect:false }},
    scales:{{
      x:{{ ticks:{{color:'#8b949e'}}, grid:{{color:'#21262d'}} }},
      y:{{ stacked:true, ticks:{{color:'#8b949e'}}, grid:{{color:'#21262d'}},
           title:{{display:true,text:'BTC Volume',color:'#8b949e'}} }},
    }},
    plugins:{{ legend:{{ labels:{{color:'#c9d1d9'}} }} }}
  }}
}});

const tbody=document.getElementById('tbody');
tableData.forEach(r=>{{
  const tr=document.createElement('tr');
  tr.innerHTML=`
    <td class="cl">${{r.bucket}}</td><td class="cl">${{r.range}}</td>
    <td>${{r.trades.toLocaleString()}}</td><td>${{r.trades_pct.toFixed(2)}}%</td><td>${{r.cum_trades_pct.toFixed(2)}}%</td>
    <td>${{r.btc.toFixed(3)}}</td><td>${{r.btc_pct.toFixed(2)}}%</td><td>${{r.cum_btc_pct.toFixed(2)}}%</td>
    <td>${{r.avg_btc.toFixed(4)}}</td><td>${{r.max_btc.toFixed(4)}}</td>
    <td class="bid">${{r.bid_trades.toLocaleString()}}</td><td class="ask">${{r.ask_trades.toLocaleString()}}</td>
    <td class="bid">${{r.bid_btc.toFixed(3)}}</td><td class="ask">${{r.ask_btc.toFixed(3)}}</td>
  `;
  tbody.appendChild(tr);
}});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Trade size distribution analysis")
    parser.add_argument("--data-root", type=Path, default=Path("./data/binance"))
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    df = load_trades(args.data_root, args.symbol, args.date)
    total_trades = len(df)
    total_btc = df["btc"].sum()

    table = build_size_table(df)
    print_table(table, args.date, total_trades, total_btc)

    percentiles = {
        "p50": df["btc"].quantile(0.5),
        "p90": df["btc"].quantile(0.9),
        "p99": df["btc"].quantile(0.99),
        "p999": df["btc"].quantile(0.999),
        "max": df["btc"].max(),
    }
    print(f"\nPERCENTILES:  P50={percentiles['p50']:.4f}  P90={percentiles['p90']:.4f}  "
          f"P99={percentiles['p99']:.4f}  P99.9={percentiles['p999']:.4f}  Max={percentiles['max']:.4f}")

    heatmap_counts = build_hourly_heatmap(df)
    heatmap_btc = build_hourly_btc_heatmap(df)

    print("\nHOURLY TRADE COUNT BY BUCKET:")
    print(heatmap_counts.to_string())

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        logger.info(f"CSV saved to {args.csv}")

    html_path = args.html or Path(f"reports/trade_sizes_{args.date}.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(generate_html(table, heatmap_counts, heatmap_btc, args.date, total_trades, total_btc, percentiles))
    logger.info(f"HTML saved to {html_path}")
    print(f"\nOpen in browser:  file://{html_path.resolve()}")


if __name__ == "__main__":
    main()

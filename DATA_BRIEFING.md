# Orderbook Data Briefing for ML Project

This document provides all necessary context to build a machine learning pipeline
for price prediction models using Binance USD-M Futures orderbook and trade data.

The data is collected by the `orderbook_checker` recorder
(https://github.com/Nils-Coding/orderbook_checker).

---

## 1. Data Source

- **Exchange**: Binance USD-M Futures
- **Symbol**: BTCUSDT (expandable to additional symbols)
- **REST API**: `https://fapi.binance.com`
- **WebSocket**: `wss://fstream.binance.com/ws`
- **Depth stream**: `@depth@100ms` (Diff-Depth, 100ms push frequency)
- **Trade stream**: `@aggTrade` (aggregated trades; older data before ~2026-03-09 may use `@trade`)
- **Orderbook depth**: 1000 levels per side (bid/ask)
- **Snapshot frequency**: Every 100ms (~10 per second, ~864,000 per day)

---

## 2. Storage Layout

```
<data_root>/
  snapshots/
    symbol=BTCUSDT/
      date=YYYY-MM-DD/
        hour=HH/
          part-0000.parquet
          part-0001.parquet
          ...
  trades/
    symbol=BTCUSDT/
      date=YYYY-MM-DD/
        hour=HH/
          part-0000.parquet
          ...
```

- Each Parquet file contains approximately 60 seconds of data
- ~60 files per hour (part-0000 through part-0059)
- Compression: ZSTD (level 3)
- All timestamps are UTC

### Data locations

| Location | Path | Content |
|---|---|---|
| GCS (primary) | `gs://orderflow-data-lake/orderbook-checker/snapshots/` | Synced from VM hourly via `gsutil rsync` |
| GCS (primary) | `gs://orderflow-data-lake/orderbook-checker/trades/` | Synced from VM hourly |
| GCS (legacy) | `gs://orderflow-data-lake/orderbook-checker/data/snapshots/` | Older data (pre-2026-01-12, different sync path) |
| Mac Mini | `/Users/schemi/data/orderbook/` | Local recording since ~2026-03-27 |

### Data availability

- GCS: 2026-01-12 through present (~80 days as of 2026-04-01)
- Mac Mini: 2026-03-27 through present (parallel backup)

---

## 3. Parquet Schemas

### 3.1 Snapshot Schema

| Column | Arrow Type | Description |
|---|---|---|
| `ts_ns` | int64 | Timestamp in nanoseconds since Unix epoch |
| `symbol` | string | Always `BTCUSDT` |
| `u` | int64 | Binance last update ID (sequence number for continuity checks) |
| `best_bid_ticks` | int64 | Best bid price in ticks |
| `best_ask_ticks` | int64 | Best ask price in ticks |
| `bids_price_delta` | list\<int32\> | 1000 values: price offset from `best_bid_ticks` (0, -1, -2, ...) |
| `bids_qty_lots` | list\<int64\> | 1000 values: quantity at each bid level |
| `asks_price_delta` | list\<int32\> | 1000 values: price offset from `best_ask_ticks` (0, +1, +2, ...) |
| `asks_qty_lots` | list\<int64\> | 1000 values: quantity at each ask level |
| `resync_epoch` | int32 | Incremented on each orderbook resynchronization |

**Delta encoding**: Prices are stored relative to the best bid/ask to save space.
To reconstruct absolute prices:

```
absolute_bid_price_ticks[i] = best_bid_ticks + bids_price_delta[i]
absolute_ask_price_ticks[i] = best_ask_ticks + asks_price_delta[i]
```

### 3.2 Trade Schema

| Column | Arrow Type | Description |
|---|---|---|
| `ts_ns` | int64 | Binance Event Time in nanoseconds since Unix epoch |
| `symbol` | string | Always `BTCUSDT` |
| `price_ticks` | int64 | Trade price in ticks |
| `qty_lots` | int64 | Trade quantity in lots |
| `is_buyer_maker` | bool | `True` = taker sells (hits bids), `False` = taker buys (lifts asks) |
| `agg_trade_id` | int64 | Aggregated trade ID (0 if from old `@trade` stream) |
| `first_trade_id` | int64 | First individual trade ID in aggregate |
| `last_trade_id` | int64 | Last individual trade ID in aggregate |

### 3.3 Unit Conversions

| Unit | Conversion | Example |
|---|---|---|
| 1 tick | 0.10 USD | `price_ticks=839650` -> $83,965.00 |
| 1 lot | 0.001 BTC | `qty_lots=5000` -> 5.000 BTC |
| `ts_ns` to seconds | `ts_ns / 1e9` | `1742900000000000000` -> `1742900000.0` |
| `ts_ns` to datetime | `datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)` | |

These scales are configured as `price_scale: 1` (1 decimal) and `qty_scale: 3`
(3 decimals) in the recorder and are specific to BTCUSDT. Other symbols would
have different tick/lot sizes.

---

## 4. Data Quality -- Known Issues and Epochs

### 4.1 Timestamp Epoch Boundary (~2026-03-24)

**Before ~2026-03-24**: Snapshot `ts_ns` uses the VM's local clock (not Binance).
A systematic offset of ~200-300ms exists between snapshot timestamps and trade
timestamps (which always use Binance Event Time). This offset fluctuates over time.

**After ~2026-03-24**: Snapshot `ts_ns` uses Binance's `E` (Event Time) field from
depth-stream WebSocket messages. Both snapshots and trades are now on the same
Binance clock. With `window=1` (matching trades to the two surrounding snapshots),
~93% of trades fall within the bid-ask spread.

**Healed data**: For dates 2026-03-15 through 2026-03-19, calibrated offsets have
been computed (10-minute sliding windows, optimal offset minimizing outside-spread
rate). Healed snapshots with adjusted timestamps exist in a separate directory
(`data/binance_healed/snapshots/`). The healing tools are:
- `tools/calibrate_timestamp_offset.py` -- computes per-window offsets
- `tools/heal_snapshots.py` -- applies offsets to create corrected files

**Recommendation for ML**: Use data from 2026-03-24 onward for training where
possible. If older data is needed, use healed snapshots and include `data_epoch`
(pre-fix vs. post-fix) as a metadata feature or stratification variable.

### 4.2 Resync Epochs

The `resync_epoch` column increments whenever the orderbook is rebuilt from a fresh
REST snapshot (after a WebSocket sequence gap or reconnection). During a resync:

- The orderbook state is discontinuous
- Snapshots before and after the resync cannot be compared directly

**Recommendation for ML**: Discard or flag any feature window that spans a
`resync_epoch` boundary. This can be detected by checking
`resync_epoch[t] != resync_epoch[t-1]` within a window.

### 4.3 Known Data Gaps

| Date | Gap | Cause |
|---|---|---|
| 2026-01-20 ~06:55 UTC | ~hours | Recorder stuck in BUFFERING (pre-recovery-fix) |
| 2026-03-09 ~23:49 UTC | Service restart | Restart for code update |
| 2026-03-23 ~17:59 UTC | Service restart | Restart for code update |
| 2026-03-26 ~08:34 UTC | ~5 hours | BUFFERING stall (sequence gap, no auto-recovery) |
| 2026-03-27 ~13:34 - 15:08 UTC | ~1.5 hours | Service restarts for deployment |
| 2026-03-31 ~09:14 UTC | ~3.5 hours | BUFFERING stall (fixed by deploying recovery loop) |

Gaps are detectable by: missing `hour=XX` directories, missing part files within
an hour, or large timestamp jumps between consecutive snapshots.

An auto-recovery mechanism was deployed on 2026-04-01 that retries sync with
exponential backoff (up to 30s), making future BUFFERING stalls self-healing.

### 4.4 Hour=01 Anomaly (Before ~2026-03-23)

On earlier dates, the `hour=01` partition sometimes has ~20 fewer or ~20 extra
Parquet files. Root cause: a daily report cronjob caused OOM (out-of-memory) on the
VM. This has been fixed by moving the report to a Cloud Run Job.

### 4.5 Trade Stream Change

- **Before ~2026-03-09**: `@trade` stream (individual trades, `agg_trade_id=0`)
- **After ~2026-03-09**: `@aggTrade` stream (aggregated, `agg_trade_id > 0`)

Aggregated trades bundle all fills at the same price within a short time window
into a single event. The total volume is identical, but the count differs.

---

## 5. Empirical Data Characteristics (from 2026-03-25 analysis)

### 5.1 Volume

- ~1.7 million aggregated trades per day
- ~165,000 BTC total daily volume
- ~864,000 orderbook snapshots per day

### 5.2 Trade Size Distribution

| Bucket | BTC Range | % of Trades | % of BTC Volume |
|---|---|---|---|
| Micro | 0.001 - 0.01 | ~66% | ~10% |
| Small | 0.01 - 0.1 | ~23% | ~21% |
| Medium | 0.1 - 1.0 | ~9% | ~30% |
| Large | 1.0 - 5.0 | ~1.5% | ~18% |
| Whale | 5.0 - 50.0 | ~0.2% | ~14% |
| Mega | > 50.0 | ~0.001% | ~5% |

- Median trade: ~0.004 BTC ($340)
- P99: ~1.7 BTC ($143k)
- Max observed: ~412 BTC ($35M)

### 5.3 Level Penetration

- **98.8%** of trades consume only Level 1 (top-of-book)
- **99.7%** consume 5 or fewer levels
- Maximum observed: 36 levels
- Implication: for many features, top-10 or even top-5 levels may suffice; deeper
  levels are relevant primarily for large trades and liquidity metrics

### 5.4 Bid-Ask Spread

- Typical spread: 1 tick (0.10 USD) -- the tightest possible
- With `window=1` matching (trade falls between two consecutive 100ms snapshots):
  ~93% of trades have prices within the min/max bid-ask spread of those two snapshots

---

## 6. ML Context from Prior Discussions

### 6.1 Prediction Target

- Probability distributions for price movements over 1-5 minute horizons
- Output: probabilities for discrete price buckets (e.g., -10 ticks, ..., +10 ticks)

### 6.2 Feature Window Strategy

- **Window size**: 5 minutes of historical data as input
- **Step size**: 15-30 seconds (overlapping windows for more training samples)
- **Samples per day**: ~2,880 (at 30s steps) to ~5,760 (at 15s steps)
- **Samples per month**: ~86k - 173k

### 6.3 Suggested Stack (discussed, not implemented)

| Component | Tool | Purpose |
|---|---|---|
| Feature engineering | Polars | Fast columnar processing on Apple Silicon |
| Training framework | PyTorch | Model training, MPS backend for M4 GPU |
| Experiment tracking | MLflow | Hyperparameters, metrics, model registry |
| Data versioning | DVC | Track data/model artifacts with Git |
| Data format | Parquet | Already in use, direct compatibility |

### 6.4 Hardware

- **Training machine**: Mac Mini M4, 24 GB RAM, 512 GB SSD
- ~3-5 GB/day compressed data; ~1 TB/year raw storage needed
- RAM constraint: cannot load full day of 1000-level snapshots into memory at once;
  process in hourly or multi-hour chunks

### 6.5 Training Data Recommendations

- **Minimum**: 2-4 weeks of clean (post-timestamp-fix) data before starting training
- **Train/test split**: chronological (not random), e.g., last 20% of days as test set
- **Validation**: separate validation set from most recent days to detect distribution shift
- **Data leakage warning**: overlapping 5-minute windows must not span train/test boundaries;
  leave a gap equal to the window size between splits

---

## 7. Code References in orderbook_checker

These files in the recorder project may be useful as reference or for import:

| File | Purpose |
|---|---|
| `src/recorder/writers.py` | Parquet schema definitions (authoritative) |
| `src/recorder/orderbook.py` | `OrderbookSnapshot` dataclass, delta encoding logic |
| `src/recorder/config.py` | `Config` dataclass with all parameters |
| `tools/diagnose_price_mismatch.py` | Trade-to-snapshot matching, spread analysis |
| `tools/analyze_level_penetration.py` | Level penetration analysis with HTML report |
| `tools/analyze_trade_sizes.py` | Trade size distribution with HTML report |
| `tools/calibrate_timestamp_offset.py` | Sliding-window offset calibration for old data |
| `tools/heal_snapshots.py` | Applies calibrated offsets to create corrected snapshots |

---

## 8. Loading Examples

### Polars (recommended for feature engineering)

```python
import polars as pl

# Load one hour of trades
trades = pl.read_parquet("data/snapshots_and_trades/trades/symbol=BTCUSDT/date=2026-03-25/hour=12/")

# Convert units
trades = trades.with_columns([
    (pl.col("price_ticks") * 0.1).alias("price_usd"),
    (pl.col("qty_lots") * 0.001).alias("qty_btc"),
    (pl.col("ts_ns") / 1e9).cast(pl.Float64).alias("ts_s"),
])
```

### PyArrow (for schema inspection or streaming)

```python
import pyarrow.parquet as pq

# Read with schema
table = pq.read_table("data/.../part-0000.parquet")
print(table.schema)

# Access nested arrays
bids_qty = table.column("bids_qty_lots")  # list<int64>
first_snapshot_bids = bids_qty[0].as_py()  # list of 1000 ints
```

### Snapshot Reconstruction

```python
import numpy as np

def reconstruct_book(row):
    """Reconstruct absolute prices from a single snapshot row."""
    bid_prices = (row["best_bid_ticks"] + np.array(row["bids_price_delta"])) * 0.1
    bid_qtys = np.array(row["bids_qty_lots"]) * 0.001
    ask_prices = (row["best_ask_ticks"] + np.array(row["asks_price_delta"])) * 0.1
    ask_qtys = np.array(row["asks_qty_lots"]) * 0.001
    return bid_prices, bid_qtys, ask_prices, ask_qtys
```

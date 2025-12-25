# Lokaler Mess-Recorder für Binance USD-M Futures  
## Top-1000 Orderbook-Vollsnapshots @100 ms + Trade-Penetration-Analyse

---

## Ziel
Ziel ist es, **empirisch zu bestimmen**, wie viele Orderbook-Levels ein späterer produktiver Recorder benötigt.  
Dazu wird lokal ein **korrekt synchronisiertes Orderbuch** geführt und **alle 100 ms ein Vollsnapshot (Top-1000 bids + asks)** gespeichert.  
Auf Basis dieser Snapshots und der Trades erfolgt anschließend eine **Trade-Penetration-Analyse**.

---

## Phase A – Mess-Recorder (lokal)

### Kernanforderungen
- **Orderbook im RAM** (Top-1000 bids/asks, konsistent)
- **Binance-konforme Synchronisation**
  - WS Diff-Depth (`<symbol>@depth@100ms`)
  - REST Snapshot (`/fapi/v1/depth?limit=1000`)
  - Sequenzprüfung (`U/u/pu`), Resync bei Gaps
- **Snapshots @100 ms** (exakt 1000 Levels je Seite, Padding)
- **Persistenz**
  - Parquet, chunked, ZSTD, keine Datei pro Tick
- **Trades mitschreiben** (`trade` oder `aggTrade`)
- **Health & Logging**
  - Reconnects, Resync-Zähler, Writer-Queue, Lag
- **Backpressure**
  - Messphase: fail-fast (keine stillen Lücken)

---

## Architektur (lokal, ein Prozess)
Asynchrone Tasks:
- WSDepthClient
- WSTradeClient
- OrderbookSync (State Machine)
- SnapshotScheduler (100 ms)
- SnapshotWriter (Queue → Parquet)
- TradeWriter (Queue → Parquet)

**Regel:** Orderbook blockiert niemals auf I/O. Writer mit bounded Queue.

---

## Orderbook-Synchronisation (Kurzfassung)

### States
`BUFFERING → FETCH_SNAPSHOT → SYNCING → LIVE`  
`RESYNCING` bei Fehlern

### Regeln
1. WS starten, Events puffern  
2. REST Snapshot laden (`lastUpdateId`)  
3. Verwerfe Events mit `u < lastUpdateId`  
4. First apply: `U ≤ lastUpdateId ≤ u`  
5. Danach strikt `pu == prev.u`, sonst Resync  
6. Apply:
   - qty absolut
   - qty == 0 → Level löschen
7. Trim auf Top-1000

---

## Datenmodell & Persistenz

### Normalisierung
- Integer-Darstellung:
  - bevorzugt `tick_size` / `step_size`
  - alternativ `price_scale` / `qty_scale`

### Snapshot-Schema (Parquet)
- ts_ns (int64)
- symbol
- u (int64)
- best_bid_ticks, best_ask_ticks
- bids_price_delta (int32[1000])
- bids_qty_lots (int64[1000])
- asks_price_delta (int32[1000])
- asks_qty_lots (int64[1000])
- optional: resync_epoch

Padding: delta=0, qty=0

### Trades-Schema
- ts_ns
- symbol
- price_ticks
- qty_lots
- is_buyer_maker

### Dateistruktur
data/
- snapshots/symbol=BTCUSDT/date=YYYY-MM-DD/hour=HH/part-0000.parquet
- trades/symbol=BTCUSDT/date=YYYY-MM-DD/hour=HH/part-0000.parquet
- logs/recorder.log

### Chunking
- Rotation alle 60–120 s
- Flush nach Zeit / Rows / Größe

---

## Konfiguration (Minimal)
symbols: ["BTCUSDT"]
snapshot_interval_ms: 100
book_depth: 1000
rest_snapshot_limit: 1000
ws_depth_stream: depth@100ms
ws_trade_stream: trade
data_root: ./data
chunk_seconds: 60
writer_queue_max: 600
fail_on_backpressure: true
price_tick_size / price_scale
qty_step_size / qty_scale
log_level: INFO

---

## Phase B – Trade-Penetration-Analyse

### Matching
- Trade → nearest previous Snapshot (`ts_snapshot ≤ ts_trade`)

### Seite
- buyer_maker = true → bids
- buyer_maker = false → asks

### Penetration
- kumulierte qty über Levels
- erstes Level mit cum ≥ trade_qty
- sonst Overflow (1001)
- optional: Preis-Impact in bps

### Output
- Anteil Trades / Notional mit Penetration > L (20,50,100,200,500,1000)
- reports/penetration_summary.parquet (+ CSV)

---

## Tests
- Unit: Apply, Snapshot, Sync
- Integration: Replay von Fixtures
- Soak: 30–60 min Live

---

## CLI
Recorder:
python -m recorder.cli record --config config.yaml

Analyse:
python tools/analyze_penetration.py --data_root ./data --symbol BTCUSDT --out ./reports

---

## Definition of Done
- Stabiler Dauerlauf
- Lückenlose Snapshots @100 ms
- Vollständige Trades
- Reproduzierbarer Penetration-Report

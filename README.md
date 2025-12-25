# Binance USD-M Futures Orderbook Recorder

Lokaler Mess-Recorder für Binance USD-M Futures mit Top-1000 Orderbook-Vollsnapshots alle 100ms und Trade-Penetration-Analyse.

## Übersicht

Dieses Tool erfasst:
- **Orderbook-Snapshots**: Alle 100ms ein vollständiger Snapshot der Top-1000 Bid/Ask-Levels
- **Trades**: Alle Trades via WebSocket
- **Persistenz**: Parquet-Dateien mit ZSTD-Kompression

Die erfassten Daten ermöglichen eine empirische Analyse der Orderbook-Penetration durch Trades.

## Installation

```bash
# Repository klonen
cd orderbook_checker

# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r requirements.txt

# Package im Entwicklungsmodus installieren
pip install -e .
```

## Konfiguration

Kopiere die Beispielkonfiguration und passe sie an:

```bash
cp config.example.yaml config.yaml
```

Wichtige Einstellungen in `config.yaml`:

```yaml
# Symbole (erweiterbar auf N Symbole)
symbols:
  - BTCUSDT

# Snapshot-Intervall
snapshot_interval_ms: 100
book_depth: 1000

# Datenverzeichnis
data_root: ./data

# Writer-Queue (Backpressure)
writer_queue_max: 600
fail_on_backpressure: true  # Bei Queue-Überlauf: Abbruch statt Datenverlust

# Preis/Mengen-Normalisierung für BTCUSDT
price_scale: 1   # 0.1 USD pro Tick
qty_scale: 3     # 0.001 BTC pro Lot
```

## Recorder starten (Phase A)

```bash
# Mit Default-Config
python -m recorder.cli record --config config.yaml

# Status prüfen
python -m recorder.cli status --config config.yaml
```

Der Recorder:
1. Verbindet sich zu Binance WebSocket (Depth + Trades)
2. Synchronisiert das Orderbook per REST-Snapshot
3. Pflegt ein lokales Top-1000 Orderbook
4. Schreibt alle 100ms einen Vollsnapshot
5. Schreibt alle Trades

### Datenstruktur

```
data/
├── snapshots/
│   └── symbol=BTCUSDT/
│       └── date=2024-01-15/
│           └── hour=14/
│               ├── part-0000.parquet
│               └── part-0001.parquet
├── trades/
│   └── symbol=BTCUSDT/
│       └── date=2024-01-15/
│           └── hour=14/
│               └── part-0000.parquet
└── logs/
    └── recorder.log
```

### Snapshot-Schema (Parquet)

| Spalte | Typ | Beschreibung |
|--------|-----|--------------|
| ts_ns | int64 | Timestamp in Nanosekunden |
| symbol | string | Trading-Symbol |
| u | int64 | Binance Update-ID |
| best_bid_ticks | int64 | Bester Bid-Preis in Ticks |
| best_ask_ticks | int64 | Bester Ask-Preis in Ticks |
| bids_price_delta | list[int32] | Preisdelta vom Best Bid (1000 Levels) |
| bids_qty_lots | list[int64] | Bid-Mengen in Lots (1000 Levels) |
| asks_price_delta | list[int32] | Preisdelta vom Best Ask (1000 Levels) |
| asks_qty_lots | list[int64] | Ask-Mengen in Lots (1000 Levels) |
| resync_epoch | int32 | Resync-Zähler |

### Trade-Schema (Parquet)

| Spalte | Typ | Beschreibung |
|--------|-----|--------------|
| ts_ns | int64 | Trade-Zeitstempel |
| symbol | string | Trading-Symbol |
| price_ticks | int64 | Preis in Ticks |
| qty_lots | int64 | Menge in Lots |
| is_buyer_maker | bool | True = Taker verkauft |

## Penetration-Analyse (Phase B)

```bash
python tools/analyze_penetration.py \
    --data_root ./data \
    --symbol BTCUSDT \
    --out ./reports \
    --csv  # Optional: auch CSV ausgeben
```

Die Analyse:
1. Lädt alle Snapshots und Trades
2. Matcht jeden Trade zum nächsten vorherigen Snapshot (ts_snapshot ≤ ts_trade)
3. Bestimmt die konsumierte Seite (is_buyer_maker → Bids, sonst Asks)
4. Berechnet das Penetration-Level (1-1000, Overflow=1001)

### Output

```
reports/
├── trade_analysis.parquet      # Alle Trades mit Penetration-Level
├── penetration_summary.parquet # Zusammenfassung nach Schwellenwerten
├── penetration_by_side.parquet # Aufschlüsselung nach Bid/Ask
└── *.csv                       # Optional CSV-Versionen
```

### Beispiel-Output

```
PENETRATION SUMMARY
============================================================
Symbol: BTCUSDT
Total trades analyzed: 125847
Total notional (lots): 45823100

Trades exceeding level thresholds:
 threshold  trades_exceeding  trades_exceeding_pct  notional_exceeding  notional_exceeding_pct
        20             12584                 10.00              892341                   19.48
        50              3421                  2.72              312456                    6.82
       100               987                  0.78              145678                    3.18
       200               234                  0.19               45678                    1.00
       500                45                  0.04               12345                    0.27
      1000                12                  0.01                3456                    0.08
      1001                 8                  0.01                2345                    0.05

PENETRATION PERCENTILES:
  50th percentile: Level 3
  90th percentile: Level 15
  99th percentile: Level 78
  Maximum: Level 1001

Overflow (>1000 levels): 0.01%
```

## Tests

```bash
# Alle Tests
pytest tests/ -v

# Nur Unit-Tests
pytest tests/test_orderbook.py tests/test_sync.py -v

# Mit Coverage
pytest tests/ --cov=src/recorder --cov-report=html
```

## Architektur

```
┌─────────────────┐     ┌─────────────────┐
│  WSDepthClient  │     │  WSTradeClient  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       │
┌─────────────────┐              │
│  OrderbookSync  │              │
│  (State Machine)│              │
└────────┬────────┘              │
         │                       │
         ▼                       │
┌─────────────────┐              │
│    Orderbook    │              │
│  (Top-1000 RAM) │              │
└────────┬────────┘              │
         │                       │
         ▼                       │
┌─────────────────┐              │
│SnapshotScheduler│              │
│   (100ms tick)  │              │
└────────┬────────┘              │
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ SnapshotWriter  │     │   TradeWriter   │
│ (Queue→Parquet) │     │ (Queue→Parquet) │
└─────────────────┘     └─────────────────┘
```

### Synchronisation (Binance-Protokoll)

1. **BUFFERING**: WS verbunden, Events werden gepuffert
2. **FETCH_SNAPSHOT**: REST-Snapshot laden (lastUpdateId)
3. **SYNCING**: Ersten gültigen Event finden (U ≤ lastUpdateId ≤ u)
4. **LIVE**: Normale Verarbeitung, Sequenzprüfung (pu == prev.u)
5. **RESYNCING**: Bei Fehlern zurück zu BUFFERING

### Backpressure

- Writer verwenden bounded Queues
- Bei Queue-Überlauf: Abbruch mit Exit-Code ≠ 0 (keine stillen Datenlücken)
- Orderbook-Task blockiert niemals auf I/O

## Health-Logging

Der Recorder loggt alle 10s:
```
HEALTH | state=LIVE | uptime=3600s | resyncs=0 | snaps=36000/36000 | snap_q=5 (0.8%) | trades=125847 | trade_q=2 (0.3%) | ws_depth=OK | ws_trade=OK
```

## Lizenz

MIT


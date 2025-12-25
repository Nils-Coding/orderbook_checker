"""Configuration management for Binance Orderbook Recorder."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class Config:
    """Recorder configuration loaded from YAML."""

    # Symbols
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])

    # Snapshot Settings
    snapshot_interval_ms: int = 100
    book_depth: int = 1000
    rest_snapshot_limit: int = 1000

    # WebSocket Streams
    ws_depth_stream: str = "depth@100ms"
    ws_trade_stream: str = "trade"

    # Data Storage
    data_root: Path = field(default_factory=lambda: Path("./data"))
    chunk_seconds: int = 60

    # Writer Queue Settings
    writer_queue_max: int = 600
    fail_on_backpressure: bool = True

    # Normalization
    price_scale: int = 1  # Dezimalstellen für Preis
    qty_scale: int = 3  # Dezimalstellen für Menge

    # Alternative normalization via tick/step size
    price_tick_size: Optional[float] = None
    qty_step_size: Optional[float] = None

    # Logging
    log_level: str = "INFO"

    # Binance API Endpoints
    binance_rest_url: str = "https://fapi.binance.com"
    binance_ws_url: str = "wss://fstream.binance.com/ws"

    # Health Monitoring
    health_log_interval_s: int = 10

    def __post_init__(self) -> None:
        """Convert data_root to Path if string."""
        if isinstance(self.data_root, str):
            self.data_root = Path(self.data_root)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle data_root conversion
        if "data_root" in data:
            data["data_root"] = Path(data["data_root"])

        return cls(**data)

    def price_to_ticks(self, price: float) -> int:
        """Convert price to integer ticks."""
        if self.price_tick_size is not None:
            return int(round(price / self.price_tick_size))
        return int(round(price * (10**self.price_scale)))

    def qty_to_lots(self, qty: float) -> int:
        """Convert quantity to integer lots."""
        if self.qty_step_size is not None:
            return int(round(qty / self.qty_step_size))
        return int(round(qty * (10**self.qty_scale)))

    def ticks_to_price(self, ticks: int) -> float:
        """Convert ticks back to price."""
        if self.price_tick_size is not None:
            return ticks * self.price_tick_size
        return ticks / (10**self.price_scale)

    def lots_to_qty(self, lots: int) -> float:
        """Convert lots back to quantity."""
        if self.qty_step_size is not None:
            return lots * self.qty_step_size
        return lots / (10**self.qty_scale)


"""Command-line interface for the Binance Orderbook Recorder."""

import asyncio
import logging
import sys
from pathlib import Path

import click

from .config import Config
from .recorder import run_recorder


def setup_logging(level: str, log_file: Path | None = None) -> None:
    """Setup logging configuration."""
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Reduce noise from external libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


@click.group()
def cli():
    """Binance USD-M Futures Orderbook Recorder."""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default="config.yaml",
    help="Path to configuration file",
)
def record(config_path: Path):
    """
    Start recording orderbook snapshots and trades.
    
    Connects to Binance USD-M Futures WebSocket streams,
    maintains synchronized orderbook, and writes 100ms snapshots
    to Parquet files.
    """
    # Load config
    try:
        config = Config.from_yaml(config_path)
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    # Setup logging
    log_file = config.data_root / "logs" / "recorder.log"
    setup_logging(config.log_level, log_file)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting recorder with config: {config_path}")
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Data root: {config.data_root}")

    # Run recorder
    try:
        exit_code = asyncio.run(run_recorder(config))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default="config.yaml",
    help="Path to configuration file",
)
def status(config_path: Path):
    """Show current data status and statistics."""
    try:
        config = Config.from_yaml(config_path)
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    click.echo(f"Data root: {config.data_root}")
    click.echo(f"Symbols: {config.symbols}")

    # Count files
    for symbol in config.symbols:
        snap_dir = config.data_root / "snapshots" / f"symbol={symbol}"
        trade_dir = config.data_root / "trades" / f"symbol={symbol}"

        snap_files = list(snap_dir.rglob("*.parquet")) if snap_dir.exists() else []
        trade_files = list(trade_dir.rglob("*.parquet")) if trade_dir.exists() else []

        click.echo(f"\n{symbol}:")
        click.echo(f"  Snapshot files: {len(snap_files)}")
        click.echo(f"  Trade files: {len(trade_files)}")


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()


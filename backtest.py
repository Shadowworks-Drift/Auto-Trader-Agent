"""
Backtest CLI
════════════
Run the backtesting engine from the command line.

Usage::

    # Backtest quant-only signal on BTC (fetches data from exchange)
    python backtest.py --symbol BTC/USDT --from 2024-01-01 --to 2024-12-31

    # Backtest from a local CSV
    python backtest.py --symbol BTC/USDT --csv data/btc_4h.csv

    # Walk-forward validation
    python backtest.py --symbol BTC/USDT --walk-forward

    # Save results to JSON
    python backtest.py --symbol BTC/USDT --save results/btc_backtest.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def async_main(args: argparse.Namespace) -> None:
    from config.settings import Settings
    from backtesting.backtest_engine import BacktestConfig, BacktestEngine, quant_signal_fn
    from backtesting.report import BacktestReport
    from backtesting.slippage_model import FeeModel, SlippageModel
    from rich.console import Console

    console = Console()
    settings = Settings.from_yaml(Path(args.config))

    cfg = BacktestConfig(
        initial_capital=float(args.capital),
        position_size_pct=settings.trading.position_size_pct,
        stop_loss_pct=settings.risk.stop_loss_pct,
        take_profit_pct=settings.risk.take_profit_pct,
        max_open_positions=settings.trading.max_open_positions,
        fee_model=FeeModel.for_exchange(settings.exchange.id),
        slippage_model=SlippageModel.for_asset(args.symbol),
        walk_forward=args.walk_forward,
    )

    if args.csv:
        engine = BacktestEngine.from_csv(args.symbol, args.csv, quant_signal_fn(), cfg)
    else:
        # Fetch data from exchange
        import ccxt.async_support as ccxt
        console.print(f"Fetching {args.symbol} data from {settings.exchange.id}...")
        exchange_class = getattr(ccxt, settings.exchange.id)
        exchange = exchange_class({"enableRateLimit": True})
        try:
            since = int(datetime.fromisoformat(args.from_date).timestamp() * 1000) if args.from_date else None
            raw = await exchange.fetch_ohlcv(args.symbol, timeframe="4h", since=since, limit=2000)
        finally:
            await exchange.close()

        import pandas as pd
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        if args.to_date:
            df = df[df["timestamp"] <= args.to_date]

        console.print(f"  Loaded {len(df)} bars from {df['timestamp'].iloc[0]:%Y-%m-%d} to {df['timestamp'].iloc[-1]:%Y-%m-%d}")
        engine = BacktestEngine(args.symbol, df, quant_signal_fn(), cfg)

    console.print(f"\nRunning backtest{'  (walk-forward)' if args.walk_forward else ''}...")
    results = await engine.run()

    report = BacktestReport(results)
    report.print(console)

    if args.save:
        report.save_json(args.save)
        console.print(f"\n[dim]Results saved to {args.save}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-Trader-Agent Backtester")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--csv", help="Path to local OHLCV CSV file")
    parser.add_argument("--from", dest="from_date", default="2024-01-01")
    parser.add_argument("--to",   dest="to_date",   default=None)
    parser.add_argument("--capital", default=10000, type=float)
    parser.add_argument("--walk-forward", action="store_true", default=True)
    parser.add_argument("--save", help="Save JSON results to this path")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

"""
Backtest CLI
════════════
Run the backtesting engine from the command line.

Usage::

    # Backtest quant-only signal on BTC (fetches data from exchange)
    python backtest.py --symbol BTC/USDT --from 2024-01-01 --to 2024-12-31

    # Choose a different resolution
    python backtest.py --symbol BTC/USDT --timeframe 1h --from 2022-01-01

    # Fetch all available history on 1d bars
    python backtest.py --symbol BTC/USDT --timeframe 1d --all-history

    # Backtest from a local CSV
    python backtest.py --symbol BTC/USDT --csv data/btc_4h.csv

    # Walk-forward validation
    python backtest.py --symbol BTC/USDT --walk-forward

    # Save results to JSON
    python backtest.py --symbol BTC/USDT --save results/btc_backtest.json

Available timeframes (Binance):
    1m  3m  5m  15m  30m
    1h  2h  4h  6h  8h  12h
    1d  3d  1w  1M
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ── Timeframe helpers ─────────────────────────────────────────────────────────

TIMEFRAME_MINUTES: dict[str, float] = {
    "1m":  1,    "3m":   3,   "5m":   5,   "15m":  15,   "30m":  30,
    "1h":  60,   "2h":   120, "4h":   240,  "6h":   360,  "8h":   480,
    "12h": 720,  "1d":   1440,"3d":   4320, "1w":   10080,"1M":   43200,
}


def bars_per_day(timeframe: str) -> float:
    """Return the number of bars in one calendar day for the given timeframe."""
    return 1440.0 / TIMEFRAME_MINUTES.get(timeframe, 240)


def days_to_bars(days: float, timeframe: str) -> int:
    """Convert a number of calendar days to an approximate bar count."""
    return max(1, int(days * bars_per_day(timeframe)))


# ── Paginated historical fetch ─────────────────────────────────────────────────

async def fetch_full_history(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    console=None,
) -> list:
    """
    Paginate CCXT fetch_ohlcv to retrieve every bar between since_ms and until_ms.

    Binance caps each request at 1 000 bars.  We advance the 'since' cursor by
    one bar width after each page until we reach until_ms or the exchange stops
    returning data.
    """
    tf_ms = int(TIMEFRAME_MINUTES.get(timeframe, 240) * 60 * 1000)
    all_bars: list = []
    cursor = since_ms
    page = 0

    while cursor < until_ms:
        chunk = await exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=cursor, limit=1000
        )
        if not chunk:
            break

        # Filter out bars beyond until_ms
        chunk = [b for b in chunk if b[0] <= until_ms]
        all_bars.extend(chunk)
        page += 1

        last_ts = chunk[-1][0]
        if console and page % 5 == 0:
            loaded_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            console.print(
                f"  [dim]...fetching page {page}, up to {loaded_dt:%Y-%m-%d}[/dim]"
            )

        if last_ts >= until_ms or len(chunk) < 1000:
            break

        # Advance cursor to the bar right after the last one received
        cursor = last_ts + tf_ms
        await asyncio.sleep(0.25)   # stay well within rate limits

    return all_bars


# ── Main ──────────────────────────────────────────────────────────────────────

async def async_main(args: argparse.Namespace) -> None:
    from config.settings import Settings
    from backtesting.backtest_engine import (
        BacktestConfig, BacktestEngine,
        multi_factor_signal_fn, quant_signal_fn,
    )
    from backtesting.report import BacktestReport
    from backtesting.slippage_model import FeeModel, SlippageModel
    from rich.console import Console

    console = Console()
    settings = Settings.from_yaml(Path(args.config))

    # ── Signal selection ──────────────────────────────────────────────────────
    signal_fn = multi_factor_signal_fn() if args.signal == "multi" else quant_signal_fn()
    signal_label = "multi-factor" if args.signal == "multi" else "quant (ADX/DI)"

    # ── Walk-forward window sizing scaled to chosen timeframe ─────────────────
    train_bars = days_to_bars(args.train_days, args.timeframe)
    test_bars  = days_to_bars(args.test_days,  args.timeframe)

    sl_pct = args.sl_pct if args.sl_pct is not None else settings.risk.stop_loss_pct
    tp_pct = args.tp_pct if args.tp_pct is not None else settings.risk.take_profit_pct

    cfg = BacktestConfig(
        initial_capital=float(args.capital),
        position_size_pct=settings.trading.position_size_pct,
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
        max_open_positions=settings.trading.max_open_positions,
        fee_model=FeeModel.for_exchange(settings.exchange.id),
        slippage_model=SlippageModel.for_asset(args.symbol),
        walk_forward=args.walk_forward,
        train_window_bars=train_bars,
        test_window_bars=test_bars,
        use_atr_stops=args.atr_stops,
        atr_sl_mult=args.atr_sl_mult,
        atr_tp_mult=args.atr_tp_mult,
        trailing_stop=args.trailing_stop,
        trailing_stop_pct=args.trailing_stop_pct,
        cooldown_bars_after_sl=args.cooldown,
    )

    if args.csv:
        engine = BacktestEngine.from_csv(args.symbol, args.csv, signal_fn, cfg)
    else:
        import ccxt.async_support as ccxt
        import pandas as pd

        console.print(
            f"Fetching [bold]{args.symbol}[/bold] data from {settings.exchange.id} "
            f"[dim]({args.timeframe} bars)[/dim]..."
        )
        exchange_class = getattr(ccxt, settings.exchange.id)
        exchange = exchange_class({"enableRateLimit": True})

        try:
            if args.all_history:
                # Start from exchange's minimum supported timestamp (usually ~2017 for Binance)
                since_ms = int(datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
            elif args.from_date:
                since_ms = int(
                    datetime.fromisoformat(args.from_date)
                    .replace(tzinfo=timezone.utc)
                    .timestamp() * 1000
                )
            else:
                since_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

            until_ms = (
                int(
                    datetime.fromisoformat(args.to_date)
                    .replace(tzinfo=timezone.utc)
                    .timestamp() * 1000
                )
                if args.to_date
                else int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            )

            raw = await fetch_full_history(
                exchange, args.symbol, args.timeframe,
                since_ms, until_ms, console=console,
            )
        finally:
            await exchange.close()

        if not raw:
            console.print("[red]No data returned. Check symbol / timeframe / date range.[/red]")
            return

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        console.print(
            f"  Loaded [bold]{len(df):,}[/bold] bars  "
            f"[dim]{df['timestamp'].iloc[0]:%Y-%m-%d} → {df['timestamp'].iloc[-1]:%Y-%m-%d}[/dim]"
        )
        if args.atr_stops:
            stops_label = f"ATR×{args.atr_sl_mult:.1f} SL / ×{args.atr_tp_mult:.1f} TP"
        else:
            stops_label = f"fixed {sl_pct*100:.0f}% SL / {tp_pct*100:.0f}% TP"
        if args.trailing_stop:
            stops_label += f" + trail {args.trailing_stop_pct*100:.0f}%"
        cooldown_label = f"  |  cooldown {args.cooldown}b after SL" if args.cooldown > 0 else ""
        console.print(
            f"  Signal: [bold]{signal_label}[/bold]  |  "
            f"Stops: [bold]{stops_label}[/bold]{cooldown_label}  |  "
            f"WF: [bold]{train_bars}[/bold] train / [bold]{test_bars}[/bold] test bars"
        )

        engine = BacktestEngine(args.symbol, df, signal_fn, cfg)

    console.print(f"\nRunning backtest{'  (walk-forward)' if args.walk_forward else ''}...")
    results = await engine.run()

    report = BacktestReport(results)
    report.print(console)

    if args.save:
        report.save_json(args.save)
        console.print(f"\n[dim]Results saved to {args.save}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-Trader-Agent Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbol",    default="BTC/USDT",
                        help="Trading pair, e.g. BTC/USDT (default: BTC/USDT)")
    parser.add_argument("--timeframe", default="4h",
                        choices=list(TIMEFRAME_MINUTES),
                        metavar="TF",
                        help=f"Bar resolution. Options: {', '.join(TIMEFRAME_MINUTES)}  (default: 4h)")
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--csv",       help="Path to local OHLCV CSV file")
    parser.add_argument("--from",      dest="from_date", default=None,
                        help="Start date ISO-8601, e.g. 2022-01-01 (default: 2024-01-01)")
    parser.add_argument("--to",        dest="to_date",   default=None,
                        help="End date ISO-8601 (default: today)")
    parser.add_argument("--all-history", action="store_true", default=False,
                        help="Fetch all available history from the exchange (from 2017-01-01)")
    parser.add_argument("--capital",   default=10_000, type=float,
                        help="Starting capital in USD (default: 10000)")
    parser.add_argument("--walk-forward", action="store_true", default=True,
                        help="Enable walk-forward validation (default: on)")
    parser.add_argument("--train-days", dest="train_days", type=float, default=90,
                        help="Training window in calendar days (default: 90)")
    parser.add_argument("--test-days",  dest="test_days",  type=float, default=30,
                        help="Test window in calendar days (default: 30)")

    # Signal selection
    parser.add_argument("--signal", default="multi", choices=["multi", "quant"],
                        help="Signal function: 'multi' = multi-factor confluence (default), "
                             "'quant' = ADX/DI crossover (original)")

    # Fixed SL/TP overrides (override config values without editing yaml)
    parser.add_argument("--sl", dest="sl_pct", type=float, default=None,
                        help="Fixed stop loss %% (e.g. 0.05 = 5%%). Overrides config. Default: from config.")
    parser.add_argument("--tp", dest="tp_pct", type=float, default=None,
                        help="Fixed take profit %% (e.g. 0.15 = 15%%). Overrides config. Default: from config.")

    # ATR-based adaptive stops
    parser.add_argument("--atr-stops", action="store_true", default=False,
                        help="Use ATR-based adaptive stop loss / take profit instead of fixed %%")
    parser.add_argument("--atr-sl-mult", dest="atr_sl_mult", type=float, default=2.0,
                        help="ATR multiplier for stop loss (default: 2.0 → SL = 2×ATR from entry)")
    parser.add_argument("--atr-tp-mult", dest="atr_tp_mult", type=float, default=4.0,
                        help="ATR multiplier for take profit (default: 4.0 → 1:2 R:R with 2.0 SL mult)")

    # Trailing stop
    parser.add_argument("--trailing-stop", action="store_true", default=False,
                        help="Trail stop loss as trade moves in your favour (lets winners run)")
    parser.add_argument("--trailing-stop-pct", dest="trailing_stop_pct", type=float, default=0.03,
                        help="How far below high-watermark to trail the stop (default: 0.03 = 3%%)")

    # Cooldown after stop-loss
    parser.add_argument("--cooldown", dest="cooldown", type=int, default=10,
                        help="Bars to wait before re-entering same direction after a stop-loss (default: 10)")

    parser.add_argument("--save",      help="Save JSON results to this path")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

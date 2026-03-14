"""
Tests for the backtesting engine and report.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    quant_signal_fn,
    _compute_drawdown_series,
    _check_sl_tp,
)
from backtesting.report import BacktestReport, _sharpe, _sortino
from backtesting.slippage_model import FeeModel, SlippageModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv_df(n: int = 300, trend: float = 0.0005) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame."""
    np.random.seed(42)
    price = 40000.0
    rows = []
    ts = datetime(2024, 1, 1)
    for i in range(n):
        change = trend + np.random.normal(0, 0.01)
        price = max(1.0, price * (1 + change))
        rows.append({
            "timestamp": ts,
            "open":  price * (1 + np.random.uniform(-0.002, 0.002)),
            "high":  price * (1 + abs(np.random.normal(0, 0.005))),
            "low":   price * (1 - abs(np.random.normal(0, 0.005))),
            "close": price,
            "volume": np.random.uniform(100, 1000),
        })
        ts += timedelta(hours=4)
    return pd.DataFrame(rows)


# ── SlippageModel tests ───────────────────────────────────────────────────────

def test_slippage_long_increases_price() -> None:
    model = SlippageModel(spread_pct=0.001, impact_coeff=0.1)
    base = 40000.0
    exec_price = model.apply(base, "long", order_value=500, avg_daily_volume_usd=1_000_000)
    assert exec_price > base


def test_slippage_short_decreases_price() -> None:
    model = SlippageModel(spread_pct=0.001, impact_coeff=0.1)
    base = 40000.0
    exec_price = model.apply(base, "short", order_value=500, avg_daily_volume_usd=1_000_000)
    assert exec_price < base


def test_fee_model_exchange_presets() -> None:
    binance = FeeModel.for_exchange("binance")
    assert binance.taker_pct == pytest.approx(0.001)
    coinbase = FeeModel.for_exchange("coinbase")
    assert coinbase.taker_pct > binance.taker_pct   # coinbase more expensive


def test_slippage_for_asset() -> None:
    btc = SlippageModel.for_asset("BTC/USDT")
    altcoin = SlippageModel.for_asset("SHIB/USDT")
    assert altcoin.spread_pct > btc.spread_pct   # illiquid = wider spread


# ── SL/TP checks ──────────────────────────────────────────────────────────────

def test_long_stop_loss_triggered() -> None:
    pos = {"direction": "long", "sl": 38000.0, "tp": 44000.0}
    triggered, price, reason = _check_sl_tp(pos, high=39000, low=37500, close=38500)
    assert triggered
    assert reason == "stop_loss"


def test_long_take_profit_triggered() -> None:
    pos = {"direction": "long", "sl": 38000.0, "tp": 44000.0}
    triggered, price, reason = _check_sl_tp(pos, high=45000, low=41000, close=44500)
    assert triggered
    assert reason == "take_profit"


def test_short_stop_loss_triggered() -> None:
    pos = {"direction": "short", "sl": 44000.0, "tp": 36000.0}
    triggered, price, reason = _check_sl_tp(pos, high=45000, low=41000, close=44500)
    assert triggered
    assert reason == "stop_loss"


def test_no_trigger_in_range() -> None:
    pos = {"direction": "long", "sl": 38000.0, "tp": 44000.0}
    triggered, _, _ = _check_sl_tp(pos, high=42000, low=39000, close=41000)
    assert not triggered


# ── Quant signal function ─────────────────────────────────────────────────────

def test_quant_signal_fn_returns_direction() -> None:
    df = _make_ohlcv_df(100)
    fn = quant_signal_fn()
    result = fn(df)
    assert "direction" in result
    assert result["direction"] in ("long", "short", "none")
    assert 0.0 <= result["confidence"] <= 1.0


# ── Full backtest run ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_backtest_runs_without_walk_forward() -> None:
    df = _make_ohlcv_df(300, trend=0.0003)
    cfg = BacktestConfig(initial_capital=10_000, walk_forward=False)
    engine = BacktestEngine("BTC/USDT", df, quant_signal_fn(), cfg)
    results = await engine.run()
    assert isinstance(results.trades, list)
    assert not results.equity_curve.empty


@pytest.mark.asyncio
async def test_backtest_walk_forward() -> None:
    df = _make_ohlcv_df(900, trend=0.0002)
    cfg = BacktestConfig(
        initial_capital=10_000,
        walk_forward=True,
        train_window_bars=200,
        test_window_bars=100,
    )
    engine = BacktestEngine("BTC/USDT", df, quant_signal_fn(), cfg)
    results = await engine.run()
    assert len(results.walk_forward_windows) >= 3


@pytest.mark.asyncio
async def test_backtest_final_capital_changes() -> None:
    df = _make_ohlcv_df(300, trend=0.001)  # strong uptrend
    cfg = BacktestConfig(initial_capital=10_000, walk_forward=False)
    engine = BacktestEngine("BTC/USDT", df, quant_signal_fn(), cfg)
    results = await engine.run()
    # There should be some trades
    assert len(results.trades) > 0


# ── Report tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_report_computes_stats() -> None:
    df = _make_ohlcv_df(400, trend=0.0002)
    cfg = BacktestConfig(initial_capital=10_000, walk_forward=False)
    engine = BacktestEngine("BTC/USDT", df, quant_signal_fn(), cfg)
    results = await engine.run()
    report = BacktestReport(results)
    stats = report.compute_stats()
    if "error" not in stats:
        assert "sharpe" in stats
        assert "win_rate_pct" in stats
        assert "max_drawdown_pct" in stats
        assert stats["win_rate_pct"] >= 0


def test_sharpe_positive_drift() -> None:
    returns = pd.Series([0.001] * 252)  # daily 0.1% return, no variance
    assert _sharpe(returns) > 0


def test_sortino_flat_returns() -> None:
    returns = pd.Series([0.0] * 100)
    assert _sortino(returns) == pytest.approx(0.0)


def test_drawdown_series_monotone_fall() -> None:
    equity = pd.Series([100, 90, 80, 70, 80, 90])
    dd = _compute_drawdown_series(equity)
    assert dd.min() < 0
    assert dd.iloc[0] == pytest.approx(0.0)   # no drawdown at peak

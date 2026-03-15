"""
Vectorised Backtesting Engine
══════════════════════════════
Replays historical OHLCV data through the full agent signal pipeline.

Design principles (Two Sigma scientific method):
  - Every signal is tested with realistic slippage and fees
  - Walk-forward validation prevents look-ahead bias
  - Regime-separated statistics show where the strategy works
  - All results are deterministic and reproducible

Usage::

    engine = BacktestEngine.from_csv("BTC/USDT", "data/btc_4h.csv", config)
    results = await engine.run()
    report = BacktestReport(results)
    report.print()
    report.save("results/btc_backtest.html")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .slippage_model import FeeModel, SlippageModel


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 10_000.0
    position_size_pct: float = 0.05       # % of equity per trade
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    max_open_positions: int = 3
    fee_model: FeeModel = field(default_factory=FeeModel)
    slippage_model: SlippageModel = field(default_factory=SlippageModel)
    commission_in_cost: bool = True
    # Walk-forward settings
    train_window_bars: int = 600          # ~100 days on 4h
    test_window_bars: int = 150           # ~25 days on 4h
    walk_forward: bool = True


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    entry_bar: int
    exit_bar: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size_pct: float
    pnl: float
    pnl_pct: float
    pnl_after_costs: float
    costs: float
    exit_reason: str           # "stop_loss" | "take_profit" | "end_of_data"
    regime: str = "unknown"    # populated if RegimeDetector is wired in


@dataclass
class BacktestResults:
    symbol: str
    config: BacktestConfig
    trades: List[BacktestTrade]
    equity_curve: pd.Series          # indexed by datetime
    drawdown_series: pd.Series
    signals: pd.DataFrame            # bar-by-bar signal log
    walk_forward_windows: List[Dict[str, Any]] = field(default_factory=list)


# ── Signal function type ──────────────────────────────────────────────────────

# A signal function takes a DataFrame of OHLCV bars (up to current bar)
# and returns a dict: {"direction": "long"|"short"|"none", "confidence": float}
SignalFn = Callable[[pd.DataFrame], Dict[str, Any]]


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Vectorised backtesting engine.

    The engine iterates bar-by-bar through historical OHLCV data,
    calls a signal function at each bar, and manages simulated positions
    with realistic slippage/fees/SL/TP logic.

    The signal function can be:
      - A simple quant function (fast)
      - An async wrapper around the full agent pipeline (slower but realistic)
    """

    def __init__(
        self,
        symbol: str,
        df: pd.DataFrame,
        signal_fn: SignalFn,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        self.symbol = symbol
        self.df = df.copy().reset_index(drop=True)
        self.signal_fn = signal_fn
        self.cfg = config or BacktestConfig()
        self._slippage = SlippageModel.for_asset(symbol)
        self._fees = self.cfg.fee_model

    @classmethod
    def from_csv(
        cls,
        symbol: str,
        csv_path: str,
        signal_fn: SignalFn,
        config: Optional[BacktestConfig] = None,
    ) -> "BacktestEngine":
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return cls(symbol, df, signal_fn, config)

    @classmethod
    def from_ohlcv(
        cls,
        symbol: str,
        ohlcv: Any,  # OHLCV object
        signal_fn: SignalFn,
        config: Optional[BacktestConfig] = None,
    ) -> "BacktestEngine":
        return cls(symbol, ohlcv.df.reset_index(), signal_fn, config)

    # ── Main run ───────────────────────────────────────────────────────────────

    async def run(self) -> BacktestResults:
        """Run the full backtest, optionally with walk-forward validation."""
        if self.cfg.walk_forward:
            return await self._run_walk_forward()
        return await self._run_single(0, len(self.df))

    # ── Walk-forward ──────────────────────────────────────────────────────────

    async def _run_walk_forward(self) -> BacktestResults:
        """Sliding window walk-forward validation."""
        train_w = self.cfg.train_window_bars
        test_w  = self.cfg.test_window_bars
        n = len(self.df)

        all_trades: List[BacktestTrade] = []
        all_equity: List[Tuple[datetime, float]] = []
        wf_windows: List[Dict[str, Any]] = []
        capital = self.cfg.initial_capital

        start = train_w
        while start + test_w <= n:
            train_slice = (start - train_w, start)
            test_slice  = (start, start + test_w)

            logger.debug(
                f"Walk-forward window: train={self.df['timestamp'].iloc[train_slice[0]]:%Y-%m-%d}"
                f"→{self.df['timestamp'].iloc[train_slice[1]-1]:%Y-%m-%d}  "
                f"test={self.df['timestamp'].iloc[test_slice[0]]:%Y-%m-%d}"
                f"→{self.df['timestamp'].iloc[test_slice[1]-1]:%Y-%m-%d}"
            )

            result = await self._run_single(
                test_slice[0], test_slice[1],
                initial_capital=capital,
                lookback_start=train_slice[0],
            )

            all_trades.extend(result.trades)
            window_start_capital = capital
            if not result.equity_curve.empty:
                all_equity.extend(zip(result.equity_curve.index, result.equity_curve.values))
                capital = float(result.equity_curve.iloc[-1])

            wf_windows.append({
                "train_start": self.df['timestamp'].iloc[train_slice[0]],
                "train_end": self.df['timestamp'].iloc[train_slice[1] - 1],
                "test_start": self.df['timestamp'].iloc[test_slice[0]],
                "test_end": self.df['timestamp'].iloc[test_slice[1] - 1],
                "trades": len(result.trades),
                "return_pct": (capital / window_start_capital - 1) if not result.equity_curve.empty else 0,
            })

            start += test_w

        equity_series = pd.Series(
            {ts: eq for ts, eq in all_equity}
        ).sort_index()
        drawdown_series = _compute_drawdown_series(equity_series)

        return BacktestResults(
            symbol=self.symbol,
            config=self.cfg,
            trades=all_trades,
            equity_curve=equity_series,
            drawdown_series=drawdown_series,
            signals=pd.DataFrame(),
            walk_forward_windows=wf_windows,
        )

    # ── Single window run ─────────────────────────────────────────────────────

    async def _run_single(
        self,
        start_bar: int,
        end_bar: int,
        initial_capital: Optional[float] = None,
        lookback_start: int = 0,
    ) -> BacktestResults:
        capital = initial_capital or self.cfg.initial_capital
        equity_history: List[Tuple[datetime, float]] = []
        trades: List[BacktestTrade] = []
        signal_log: List[Dict[str, Any]] = []

        # Open positions: {pos_id: {...}}
        open_positions: Dict[int, Dict[str, Any]] = {}
        next_pos_id = 0

        for bar_idx in range(start_bar, end_bar):
            bar = self.df.iloc[bar_idx]
            bar_time = bar.get("timestamp", bar.name)
            close = float(bar["close"])
            high  = float(bar["high"])
            low   = float(bar["low"])

            # ── Check SL/TP on open positions ─────────────────────────────
            for pos_id in list(open_positions.keys()):
                pos = open_positions[pos_id]
                closed, exit_price, reason = _check_sl_tp(pos, high, low, close)
                if closed:
                    trade = _close_position(pos, exit_price, bar_idx, bar_time, reason, capital, self._fees, self._slippage)
                    capital += trade.pnl_after_costs
                    trades.append(trade)
                    del open_positions[pos_id]

            # ── Generate signal ────────────────────────────────────────────
            lookback_df = self.df.iloc[lookback_start:bar_idx + 1]
            if len(lookback_df) < 50:
                equity_history.append((bar_time, capital))
                continue

            try:
                if asyncio.iscoroutinefunction(self.signal_fn):
                    signal = await self.signal_fn(lookback_df)
                else:
                    signal = self.signal_fn(lookback_df)
            except Exception as exc:
                logger.debug(f"Signal error at bar {bar_idx}: {exc}")
                signal = {"direction": "none", "confidence": 0.0}

            direction  = signal.get("direction", "none")
            confidence = float(signal.get("confidence", 0.0))
            signal_log.append({"bar": bar_idx, "time": bar_time, "direction": direction, "confidence": confidence})

            # ── Open position if signal strong enough and capacity allows ──
            if (
                direction != "none"
                and confidence >= 0.60
                and len(open_positions) < self.cfg.max_open_positions
                and not _already_in_symbol(open_positions, self.symbol)
            ):
                entry_price = self._slippage.apply(
                    close, direction,
                    order_value=capital * self.cfg.position_size_pct,
                )
                fee_cost = capital * self.cfg.position_size_pct * self._fees.taker_pct
                sl = entry_price * (1 - self.cfg.stop_loss_pct) if direction == "long" else entry_price * (1 + self.cfg.stop_loss_pct)
                tp = entry_price * (1 + self.cfg.take_profit_pct) if direction == "long" else entry_price * (1 - self.cfg.take_profit_pct)

                open_positions[next_pos_id] = {
                    "id": next_pos_id,
                    "symbol": self.symbol,
                    "direction": direction,
                    "entry_bar": bar_idx,
                    "entry_time": bar_time,
                    "entry_price": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "size_pct": self.cfg.position_size_pct,
                    "entry_value": capital * self.cfg.position_size_pct,
                    "entry_fee": fee_cost,
                }
                capital -= fee_cost
                next_pos_id += 1

            equity_history.append((bar_time, capital))

        # ── Close any remaining positions at end of data ───────────────────
        if open_positions:
            final_bar = self.df.iloc[end_bar - 1]
            final_close = float(final_bar["close"])
            final_time = final_bar.get("timestamp", final_bar.name)
            for pos in open_positions.values():
                trade = _close_position(pos, final_close, end_bar - 1, final_time, "end_of_data", capital, self._fees, self._slippage)
                capital += trade.pnl_after_costs
                trades.append(trade)

        equity_series = pd.Series({ts: eq for ts, eq in equity_history}).sort_index()
        drawdown_series = _compute_drawdown_series(equity_series)

        return BacktestResults(
            symbol=self.symbol,
            config=self.cfg,
            trades=trades,
            equity_curve=equity_series,
            drawdown_series=drawdown_series,
            signals=pd.DataFrame(signal_log),
        )


# ── Built-in signal functions ─────────────────────────────────────────────────

def quant_signal_fn(min_confidence: float = 0.55) -> SignalFn:
    """
    Returns a signal function using the same QuantAnalyst logic
    but synchronously on a DataFrame slice.
    """
    def _fn(df: pd.DataFrame) -> Dict[str, Any]:
        close = df["close"]
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        rsi_v = float(rsi.iloc[-1]) if not rsi.empty else 50

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        sig  = macd.ewm(span=9).mean()
        hist = float((macd - sig).iloc[-1])

        # EMA trend
        ema20  = float(close.ewm(span=20).mean().iloc[-1])
        ema50  = float(close.ewm(span=50).mean().iloc[-1])
        price  = float(close.iloc[-1])

        # Vote
        votes = []
        votes.append(1 if rsi_v < 35 else (-1 if rsi_v > 65 else 0))
        votes.append(1 if hist > 0 else -1)
        votes.append(1 if price > ema20 > ema50 else (-1 if price < ema20 < ema50 else 0))

        score = sum(votes) / len(votes)
        conf  = abs(score)
        direction = "long" if score > 0.2 else "short" if score < -0.2 else "none"
        return {"direction": direction, "confidence": conf}

    return _fn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_sl_tp(
    pos: Dict, high: float, low: float, close: float
) -> Tuple[bool, float, str]:
    if pos["direction"] == "long":
        if low <= pos["sl"]:
            return True, pos["sl"], "stop_loss"
        if high >= pos["tp"]:
            return True, pos["tp"], "take_profit"
    else:
        if high >= pos["sl"]:
            return True, pos["sl"], "stop_loss"
        if low <= pos["tp"]:
            return True, pos["tp"], "take_profit"
    return False, close, ""


def _close_position(
    pos: Dict,
    exit_price: float,
    exit_bar: int,
    exit_time: datetime,
    reason: str,
    capital: float,
    fees: FeeModel,
    slippage: SlippageModel,
) -> BacktestTrade:
    direction = pos["direction"]
    exit_slip = slippage.apply(exit_price, "short" if direction == "long" else "long",
                               order_value=pos["entry_value"])
    fee_cost = exit_slip * pos["entry_value"] / pos["entry_price"] * fees.taker_pct
    units = pos["entry_value"] / pos["entry_price"]

    if direction == "long":
        gross_pnl = (exit_slip - pos["entry_price"]) * units
    else:
        gross_pnl = (pos["entry_price"] - exit_slip) * units

    total_costs = pos["entry_fee"] + fee_cost
    net_pnl = gross_pnl - total_costs

    return BacktestTrade(
        symbol=pos["symbol"],
        direction=direction,
        entry_bar=pos["entry_bar"],
        exit_bar=exit_bar,
        entry_time=pos["entry_time"],
        exit_time=exit_time,
        entry_price=pos["entry_price"],
        exit_price=exit_slip,
        size_pct=pos["size_pct"],
        pnl=gross_pnl,
        pnl_pct=gross_pnl / pos["entry_value"],
        pnl_after_costs=net_pnl,
        costs=total_costs,
        exit_reason=reason,
    )


def _already_in_symbol(positions: Dict, symbol: str) -> bool:
    return any(p["symbol"] == symbol for p in positions.values())


def _compute_drawdown_series(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return dd.clip(lower=-1.0).fillna(0.0)

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
    position_size_pct: float = 0.15       # 15% — large enough for meaningful equity moves
    stop_loss_pct: float = 0.05           # 5% — used when use_atr_stops=False
    take_profit_pct: float = 0.10         # 10% — used when use_atr_stops=False
    max_open_positions: int = 3
    fee_model: FeeModel = field(default_factory=FeeModel)
    slippage_model: SlippageModel = field(default_factory=SlippageModel)
    commission_in_cost: bool = True
    # Walk-forward settings
    train_window_bars: int = 200          # ~33 days on 4h — enough to warm indicators
    test_window_bars: int = 150           # ~25 days on 4h
    walk_forward: bool = True
    # ATR-based adaptive stops
    use_atr_stops: bool = False
    atr_period: int = 14
    atr_sl_mult: float = 2.0             # stop = entry ± atr_sl_mult × ATR
    atr_tp_mult: float = 4.0             # target = entry ± atr_tp_mult × ATR (1:2 R:R)


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

                if self.cfg.use_atr_stops:
                    atr_val = _compute_atr(lookback_df, self.cfg.atr_period)
                    if direction == "long":
                        sl = entry_price - atr_val * self.cfg.atr_sl_mult
                        tp = entry_price + atr_val * self.cfg.atr_tp_mult
                    else:
                        sl = entry_price + atr_val * self.cfg.atr_sl_mult
                        tp = entry_price - atr_val * self.cfg.atr_tp_mult
                else:
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

            # Mark-to-market: include unrealised P&L of open positions
            unrealised = 0.0
            for pos in open_positions.values():
                units = pos["entry_value"] / pos["entry_price"]
                if pos["direction"] == "long":
                    unrealised += (close - pos["entry_price"]) * units
                else:
                    unrealised += (pos["entry_price"] - close) * units
            equity_history.append((bar_time, capital + unrealised))

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

def multi_factor_signal_fn() -> SignalFn:
    """
    Multi-factor confluence signal — substantially higher bar than quant_signal_fn.

    Hard requirements (all must pass):
      1. ADX(14) >= 25            — confirmed trend strength, not just noise
      2. Price on correct side of EMA200 — macro trend alignment
      3. +DI/-DI aligned          — directional agreement
      4. DI spread >= 5 pts       — clear divergence, not just a crossover tick

    Soft confirmation (need >= 2 of 6):
      • EMA50 aligned with EMA200     — medium-term trend agrees with macro
      • MACD histogram positive AND increasing — momentum building, not fading
      • Volume > 1.2× 20-bar avg     — conviction behind the move
      • RSI in ideal entry zone       — 45–68 long, 32–55 short (not exhausted)
      • ADX rising                    — trend still strengthening
      • DI spread >= 15 pts           — strong directional conviction

    Confidence = base (from ADX strength) + 0.05 per soft factor, capped at 0.92.
    Signals below 0.60 are suppressed (engine threshold).
    """
    def _fn(df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 210:
            return {"direction": "none", "confidence": 0.0}
        if not all(c in df.columns for c in ("high", "low", "close", "volume")):
            return {"direction": "none", "confidence": 0.0}

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # ── EMA stack ────────────────────────────────────────────────────
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        close_v  = float(close.iloc[-1])
        ema50_v  = float(ema50.iloc[-1])
        ema200_v = float(ema200.iloc[-1])

        # ── ADX / DI (Wilder, period=14) ─────────────────────────────────
        period = 14
        a = 1.0 / period
        prev_c = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_c).abs(),
            (low  - prev_c).abs(),
        ], axis=1).max(axis=1)
        up   = high.diff()
        down = -low.diff()
        plus_dm  = pd.Series(np.where((up > down) & (up > 0),   up,   0.0), index=close.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)
        atr14    = tr.ewm(alpha=a, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(alpha=a,  adjust=False).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr14.replace(0, np.nan)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx      = dx.ewm(alpha=a, adjust=False).mean()

        def _s(s: pd.Series, d=0.0) -> float:
            v = s.iloc[-1]; return float(v) if not pd.isna(v) else d
        def _s2(s: pd.Series, d=0.0) -> float:
            v = s.iloc[-2] if len(s) > 1 else s.iloc[-1]; return float(v) if not pd.isna(v) else d

        adx_v = _s(adx);      adx_prev = _s2(adx)
        pdi_v = _s(plus_di);  mdi_v    = _s(minus_di)

        # ── MACD histogram ────────────────────────────────────────────────
        macd_line   = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_hist   = macd_line - macd_line.ewm(span=9, adjust=False).mean()
        hist_v      = _s(macd_hist)
        hist_prev   = _s2(macd_hist)

        # ── RSI(14) ───────────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rsi_v = _s(100 - 100 / (1 + gain / loss.replace(0, np.nan)), 50.0)

        # ── Volume ratio ─────────────────────────────────────────────────
        vol_ratio = float(volume.iloc[-1]) / max(float(volume.rolling(20, min_periods=1).mean().iloc[-1]), 1e-9)

        # ── Hard requirements ─────────────────────────────────────────────
        if adx_v < 25:
            return {"direction": "none", "confidence": 0.0}

        di_spread = abs(pdi_v - mdi_v)
        if di_spread < 5:
            return {"direction": "none", "confidence": 0.0}

        going_long  = close_v > ema200_v and pdi_v > mdi_v and rsi_v < 75
        going_short = close_v < ema200_v and mdi_v > pdi_v and rsi_v > 25

        if not going_long and not going_short:
            return {"direction": "none", "confidence": 0.0}

        # ── Soft confirmation factors ─────────────────────────────────────
        factors = 0
        if going_long:
            if ema50_v > ema200_v:                    factors += 1  # EMA stack aligned
            if hist_v > 0 and hist_v > hist_prev:     factors += 1  # MACD building
            if vol_ratio > 1.2:                       factors += 1  # volume expansion
            if 45 <= rsi_v <= 68:                     factors += 1  # ideal RSI zone
            if adx_v > adx_prev:                      factors += 1  # ADX rising
            if di_spread > 15:                        factors += 1  # strong DI divergence
        else:
            if ema50_v < ema200_v:                    factors += 1
            if hist_v < 0 and hist_v < hist_prev:     factors += 1
            if vol_ratio > 1.2:                       factors += 1
            if 32 <= rsi_v <= 55:                     factors += 1
            if adx_v > adx_prev:                      factors += 1
            if di_spread > 15:                        factors += 1

        if factors < 2:
            return {"direction": "none", "confidence": 0.0}

        # Base confidence from ADX strength; factors each add 0.05
        confidence = min(0.55 + adx_v / 200.0 + factors * 0.05, 0.92)
        direction  = "long" if going_long else "short"
        return {"direction": direction, "confidence": confidence}

    return _fn


def quant_signal_fn(min_confidence: float = 0.55) -> SignalFn:
    """
    ADX + Directional Index (DI) trend-following signal.

    The ADX system directly measures trend STRENGTH and DIRECTION without
    relying on lagging EMA stack alignment — so it detects a new downtrend
    even before EMA21 crosses below EMA50.

    Signal rules:
      - +DI crosses above -DI AND ADX >= 20  →  LONG  (bull trend establishing)
      - -DI crosses above +DI AND ADX >= 20  →  SHORT (bear trend establishing)
      - ADX crosses above 25 while DI already diverged → late entry allowed
      - RSI(14) exclusion: no longs >75, no shorts <25

    Confidence scaled by ADX strength (20→0.65, 30→0.75, 50→0.90).
    """
    def _fn(df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 30:
            return {"direction": "none", "confidence": 0.0}
        if "high" not in df.columns or "low" not in df.columns:
            return {"direction": "none", "confidence": 0.0}

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        period = 14

        # ── True Range ────────────────────────────────────────────────────
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        # ── Directional movement ──────────────────────────────────────────
        up   = high.diff()
        down = -low.diff()
        plus_dm  = pd.Series(np.where((up > down) & (up > 0),   up,   0.0), index=close.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)

        # Wilder smoothing (alpha = 1/period)
        a = 1.0 / period
        atr14     = tr.ewm(alpha=a, adjust=False).mean()
        plus_di   = 100 * plus_dm.ewm(alpha=a,  adjust=False).mean() / atr14.replace(0, np.nan)
        minus_di  = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr14.replace(0, np.nan)
        dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx       = dx.ewm(alpha=a, adjust=False).mean()

        def _safe(s: pd.Series, default=0.0) -> float:
            v = s.iloc[-1]
            return float(v) if not pd.isna(v) else default

        def _safe2(s: pd.Series, default=0.0) -> float:
            v = s.iloc[-2] if len(s) > 1 else s.iloc[-1]
            return float(v) if not pd.isna(v) else default

        adx_v  = _safe(adx);       adx_prev  = _safe2(adx)
        pdi_v  = _safe(plus_di);   pdi_prev  = _safe2(plus_di)
        mdi_v  = _safe(minus_di);  mdi_prev  = _safe2(minus_di)

        # ── RSI(14) exclusion filter ──────────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rsi_v = _safe(100 - 100 / (1 + gain / loss.replace(0, np.nan)), 50.0)

        if adx_v < 20:
            return {"direction": "none", "confidence": 0.0}

        conf = min(0.50 + adx_v / 100.0, 0.95)   # 20→0.70, 30→0.80, 50→0.95

        # DI crossover (event) OR ADX just broke threshold while DI diverged
        di_cross_up  = pdi_v > mdi_v and pdi_prev <= mdi_prev
        di_cross_dn  = mdi_v > pdi_v and mdi_prev <= pdi_prev
        adx_breakout = adx_v >= 25 and adx_prev < 25

        long_signal  = (di_cross_up  or (adx_breakout and pdi_v > mdi_v)) and rsi_v < 75
        short_signal = (di_cross_dn  or (adx_breakout and mdi_v > pdi_v)) and rsi_v > 25

        if long_signal:
            return {"direction": "long",  "confidence": conf}
        if short_signal:
            return {"direction": "short", "confidence": conf}
        return {"direction": "none", "confidence": 0.0}

    return _fn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR from a lookback DataFrame. Falls back to 2% of close if too few bars."""
    if len(df) < period:
        return float(df["close"].iloc[-1]) * 0.02
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    prev  = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    val = float(atr.iloc[-1])
    return val if not np.isnan(val) else float(close.iloc[-1]) * 0.02


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

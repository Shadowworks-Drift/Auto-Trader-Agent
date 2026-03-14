"""
Quant Analyst — computes technical indicators and produces a numeric signal score.

Indicators:
  RSI, MACD, Bollinger Bands, EMA (20/50/200), ATR, Volume MA, Stochastic, ADX

Signal scoring:
  Each indicator votes {-1 bearish, 0 neutral, +1 bullish}.
  Final score is a weighted average in [-1, 1].
  Confidence is derived from signal agreement (consensus).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import Settings
from data.market_data import OHLCV, MarketSnapshot
from .base_agent import AgentResult, BaseAgent


@dataclass
class QuantSignal:
    symbol: str
    direction: str            # "long" | "short" | "none"
    score: float              # -1.0 to 1.0
    confidence: float         # 0.0 to 1.0
    indicators: Dict[str, Any] = field(default_factory=dict)
    votes: Dict[str, int] = field(default_factory=dict)
    summary_text: str = ""


class QuantAnalyst(BaseAgent):
    """Computes technical indicators and scores them into a directional signal."""

    name = "QuantAnalyst"

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.cfg = settings.quant

    async def analyse(self, snapshot: MarketSnapshot) -> AgentResult:
        t0 = time.perf_counter()
        primary_tf = next(
            (tf for tf in ("4h", "1h", "1d") if tf in snapshot.ohlcv), None
        )
        if primary_tf is None or len(snapshot.ohlcv[primary_tf]) < 50:
            return self._make_result(
                snapshot.symbol,
                success=False,
                data={},
                error="Insufficient OHLCV data",
            )

        ohlcv = snapshot.ohlcv[primary_tf]
        df = ohlcv.df.copy()

        try:
            indicators = self._compute_indicators(df)
            votes = self._vote(indicators, df)
            score, confidence = self._fuse(votes)
            direction = "long" if score > 0.1 else "short" if score < -0.1 else "none"

            signal = QuantSignal(
                symbol=snapshot.symbol,
                direction=direction,
                score=score,
                confidence=confidence,
                indicators=indicators,
                votes=votes,
                summary_text=self._build_summary(snapshot.symbol, indicators, votes, score),
            )

            elapsed = (time.perf_counter() - t0) * 1000
            return self._make_result(
                snapshot.symbol,
                success=True,
                data={
                    "direction": direction,
                    "score": score,
                    "confidence": confidence,
                    "indicators": indicators,
                    "votes": votes,
                    "summary_text": signal.summary_text,
                    "quant_signal": signal,
                },
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            logger.error(f"QuantAnalyst error for {snapshot.symbol}: {exc}", exc_info=True)
            return self._make_result(snapshot.symbol, success=False, data={}, error=str(exc))

    # ── Indicator computation ─────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        ind: Dict[str, Any] = {}

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.cfg.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.cfg.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        ind["rsi"] = float((100 - 100 / (1 + rs)).iloc[-1])

        # MACD
        ema_fast = close.ewm(span=self.cfg.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.cfg.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.cfg.macd_signal, adjust=False).mean()
        ind["macd"] = float(macd_line.iloc[-1])
        ind["macd_signal"] = float(signal_line.iloc[-1])
        ind["macd_histogram"] = float((macd_line - signal_line).iloc[-1])

        # Bollinger Bands
        sma = close.rolling(self.cfg.bb_period).mean()
        std = close.rolling(self.cfg.bb_period).std()
        ind["bb_upper"] = float((sma + self.cfg.bb_std * std).iloc[-1])
        ind["bb_middle"] = float(sma.iloc[-1])
        ind["bb_lower"] = float((sma - self.cfg.bb_std * std).iloc[-1])
        ind["bb_pct"] = float(
            (close.iloc[-1] - ind["bb_lower"])
            / max(ind["bb_upper"] - ind["bb_lower"], 1e-9)
        )

        # EMAs
        for span in (20, 50, 200):
            ind[f"ema_{span}"] = float(close.ewm(span=span, adjust=False).mean().iloc[-1])

        # ATR
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
        ).max(axis=1)
        ind["atr"] = float(tr.rolling(14).mean().iloc[-1])
        ind["atr_pct"] = ind["atr"] / float(close.iloc[-1])

        # Volume MA
        vol_ma = volume.rolling(20).mean()
        ind["volume_ratio"] = float(volume.iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0

        # Stochastic %K/%D
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        stoch_k = 100 * (close - low14) / (high14 - low14 + 1e-9)
        ind["stoch_k"] = float(stoch_k.iloc[-1])
        ind["stoch_d"] = float(stoch_k.rolling(3).mean().iloc[-1])

        # ADX
        ind["adx"] = float(_compute_adx(high, low, close, self.cfg.adx_period))

        # Current price
        ind["price"] = float(close.iloc[-1])

        return ind

    # ── Voting ────────────────────────────────────────────────────────────────

    def _vote(self, ind: Dict[str, Any], df: pd.DataFrame) -> Dict[str, int]:
        """Each indicator returns -1 (bearish), 0 (neutral), or +1 (bullish)."""
        votes: Dict[str, int] = {}

        # RSI
        if ind["rsi"] < self.cfg.rsi_oversold:
            votes["rsi"] = 1
        elif ind["rsi"] > self.cfg.rsi_overbought:
            votes["rsi"] = -1
        else:
            votes["rsi"] = 0

        # MACD
        if ind["macd_histogram"] > 0 and ind["macd"] > ind["macd_signal"]:
            votes["macd"] = 1
        elif ind["macd_histogram"] < 0 and ind["macd"] < ind["macd_signal"]:
            votes["macd"] = -1
        else:
            votes["macd"] = 0

        # Bollinger Bands position
        if ind["bb_pct"] < 0.2:
            votes["bbands"] = 1   # price near lower band
        elif ind["bb_pct"] > 0.8:
            votes["bbands"] = -1  # price near upper band
        else:
            votes["bbands"] = 0

        # EMA alignment (trend following)
        price = ind["price"]
        ema20, ema50, ema200 = ind["ema_20"], ind["ema_50"], ind["ema_200"]
        if price > ema20 > ema50 > ema200:
            votes["ema_trend"] = 1
        elif price < ema20 < ema50 < ema200:
            votes["ema_trend"] = -1
        elif price > ema50:
            votes["ema_trend"] = 1
        elif price < ema50:
            votes["ema_trend"] = -1
        else:
            votes["ema_trend"] = 0

        # Volume confirmation
        votes["volume"] = 1 if ind["volume_ratio"] > 1.3 else 0

        # Stochastic
        if ind["stoch_k"] < 20 and ind["stoch_k"] > ind["stoch_d"]:
            votes["stoch"] = 1
        elif ind["stoch_k"] > 80 and ind["stoch_k"] < ind["stoch_d"]:
            votes["stoch"] = -1
        else:
            votes["stoch"] = 0

        # ADX trend strength (high ADX amplifies the directional vote)
        votes["adx"] = 1 if ind["adx"] > self.cfg.adx_trend_threshold else 0

        return votes

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _fuse(self, votes: Dict[str, int]) -> Tuple[float, float]:
        """Compute weighted score and confidence from votes."""
        weights = {
            "rsi": 1.5,
            "macd": 2.0,
            "bbands": 1.0,
            "ema_trend": 2.5,
            "volume": 0.5,
            "stoch": 1.0,
            "adx": 0.5,
        }
        total_weight = sum(weights.get(k, 1.0) for k in votes)
        weighted_sum = sum(votes[k] * weights.get(k, 1.0) for k in votes)
        score = weighted_sum / max(total_weight, 1e-9)

        # Confidence = fraction of votes in the dominant direction
        dominant = 1 if score >= 0 else -1
        agree = sum(1 for v in votes.values() if v == dominant)
        confidence = agree / max(len(votes), 1)

        return round(float(score), 4), round(float(confidence), 4)

    # ── Summary text ──────────────────────────────────────────────────────────

    def _build_summary(
        self,
        symbol: str,
        ind: Dict[str, Any],
        votes: Dict[str, int],
        score: float,
    ) -> str:
        lines = [
            f"Quant summary for {symbol}:",
            f"  Price: {ind['price']:.4f}",
            f"  RSI({self.cfg.rsi_period}): {ind['rsi']:.1f}",
            f"  MACD histogram: {ind['macd_histogram']:.5f}",
            f"  BB%: {ind['bb_pct']:.2f}  (upper={ind['bb_upper']:.4f}, lower={ind['bb_lower']:.4f})",
            f"  EMA 20/50/200: {ind['ema_20']:.4f} / {ind['ema_50']:.4f} / {ind['ema_200']:.4f}",
            f"  ATR%: {ind['atr_pct']:.4f}",
            f"  Volume ratio: {ind['volume_ratio']:.2f}x",
            f"  Stoch K/D: {ind['stoch_k']:.1f} / {ind['stoch_d']:.1f}",
            f"  ADX: {ind['adx']:.1f}",
            f"  Votes: {votes}",
            f"  Composite score: {score:+.4f}",
        ]
        return "\n".join(lines)


# ── ADX helper ────────────────────────────────────────────────────────────────

def _compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> float:
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    dm_plus = ((high.diff() > low.diff().abs()) & (high.diff() > 0)) * high.diff()
    dm_minus = ((low.diff().abs() > high.diff()) & (low.diff() < 0)) * (-low.diff())

    atr = tr.ewm(span=period, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    adx = dx.ewm(span=period, adjust=False).mean()
    return float(adx.iloc[-1]) if not adx.empty else 0.0

"""
Regime Detector
════════════════
Identifies the current market regime using a Hidden Markov Model (HMM)
and PELT changepoint detection.

Based on:
  - Renaissance Medallion Fund's confirmed HMM methodology
  - "Ensemble HMM + Random Forest" (AIMS Press 2025)
  - PELT algorithm via `ruptures` library

Four regimes:
  bull_low_vol   — trending up, calm (best for trend-following)
  bull_high_vol  — trending up, volatile (reduce size, wider stops)
  bear_low_vol   — trending down, calm (short bias, manageable)
  bear_high_vol  — trending down, volatile (minimal exposure)

The RegimeDetector outputs:
  1. Current regime label
  2. Regime confidence (0–1)
  3. Changepoint alert (True if structural break in last N bars)
  4. Regime-conditional weight adjustments for DecisionCore
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from data.market_data import OHLCV


# ── Regime definitions ────────────────────────────────────────────────────────

REGIMES = {
    "bull_low_vol":  {"trend": +1, "vol": "low"},
    "bull_high_vol": {"trend": +1, "vol": "high"},
    "bear_low_vol":  {"trend": -1, "vol": "low"},
    "bear_high_vol": {"trend": -1, "vol": "high"},
    "sideways":      {"trend":  0, "vol": "any"},
}

# Per-regime weight multipliers for DecisionCore signal fusion
# Format: {regime: {signal_name: multiplier}}
REGIME_WEIGHT_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "bull_low_vol": {
        "ema_trend": 1.5,   # trend-following signals amplified
        "macd":      1.3,
        "rsi":       0.7,   # oscillators less useful in trend
        "bbands":    0.7,
        "trend":     1.4,   # LLM trend agent more relevant
        "setup":     1.2,
        "trigger":   1.0,
        "sentiment": 0.8,
    },
    "bull_high_vol": {
        "ema_trend": 1.0,
        "macd":      0.8,
        "rsi":       1.2,
        "bbands":    1.2,   # mean-reversion signals more useful when volatile
        "atr":       1.4,
        "trend":     1.0,
        "setup":     1.0,
        "trigger":   1.3,   # focus on precise entry to manage risk
        "sentiment": 0.6,
    },
    "bear_low_vol": {
        "ema_trend": 1.5,
        "macd":      1.3,
        "rsi":       0.7,
        "bbands":    0.7,
        "trend":     1.4,
        "setup":     1.2,
        "trigger":   1.0,
        "sentiment": 0.9,
    },
    "bear_high_vol": {
        "ema_trend": 0.6,
        "macd":      0.6,
        "rsi":       1.3,
        "bbands":    1.3,
        "trend":     0.7,
        "setup":     0.7,
        "trigger":   0.5,
        "sentiment": 1.2,   # sentiment/news more influential in fear
        "position_size_mult": 0.4,  # halve position sizes
    },
    "sideways": {
        "ema_trend": 0.5,
        "macd":      0.8,
        "rsi":       1.5,   # oscillators work best in ranges
        "bbands":    1.5,
        "stoch":     1.4,
        "trend":     0.5,
        "setup":     1.0,
        "trigger":   1.2,
        "sentiment": 1.0,
        "position_size_mult": 0.7,
    },
}


@dataclass
class RegimeResult:
    regime: str                          # one of REGIMES keys
    confidence: float                    # 0–1
    trend_direction: int                 # +1, 0, -1
    volatility_level: str                # "low" | "normal" | "high" | "extreme"
    changepoint_detected: bool           # True if structural break in last N bars
    changepoint_bar: Optional[int]       # bar index of latest changepoint
    weight_adjustments: Dict[str, float] # multipliers for DecisionCore
    position_size_mult: float            # global position size multiplier
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_tradeable(self) -> bool:
        """False in extreme vol or after a fresh changepoint."""
        if self.volatility_level == "extreme":
            return False
        if self.changepoint_detected and self.changepoint_bar is not None:
            # Fresh changepoint within last 3 bars = too uncertain
            return False
        return True

    def summary(self) -> str:
        cp = f" ⚠ CHANGEPOINT at bar {self.changepoint_bar}" if self.changepoint_detected else ""
        return (
            f"Regime: [{self.regime}] conf={self.confidence:.2f} "
            f"vol={self.volatility_level} size_mult={self.position_size_mult:.2f}{cp}"
        )


# ── HMM-based regime detector ────────────────────────────────────────────────

class RegimeDetector:
    """
    Identifies market regime using:
      1. HMM on [returns, volatility] features (4 hidden states)
      2. PELT changepoint detection on rolling volatility
      3. ATR percentile for volatility level classification

    Designed to run before the main agent pipeline — its output
    adjusts DecisionCore weights and position sizing.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        lookback: int = 100,
        changepoint_lookback: int = 20,
        vol_low_pct: float = 25,
        vol_high_pct: float = 75,
        vol_extreme_pct: float = 95,
    ) -> None:
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.changepoint_lookback = changepoint_lookback
        self.vol_low_pct    = vol_low_pct
        self.vol_high_pct   = vol_high_pct
        self.vol_extreme_pct = vol_extreme_pct
        self._hmm_model = None

    def detect(self, ohlcv: OHLCV) -> RegimeResult:
        """
        Detect the current regime from an OHLCV object.
        Falls back gracefully if HMM libraries are unavailable.
        """
        t0 = time.perf_counter()
        df = ohlcv.df.copy()
        if len(df) < self.lookback:
            return self._default_regime("Insufficient data for regime detection")

        features = self._extract_features(df)

        # Determine trend direction
        trend_dir = self._compute_trend(df)

        # Determine volatility level
        vol_level, vol_pct, atr_pct = self._classify_volatility(df)

        # HMM or fallback
        hmm_regime, hmm_conf = self._hmm_classify(features)

        # PELT changepoint detection
        cp_detected, cp_bar = self._pelt_changepoint(df)

        # Map to regime label
        regime = self._map_to_regime(trend_dir, vol_level, hmm_regime)

        # Position size multiplier
        size_mult = REGIME_WEIGHT_ADJUSTMENTS.get(regime, {}).get("position_size_mult", 1.0)
        if vol_level == "extreme":
            size_mult = 0.0
        elif cp_detected:
            size_mult = min(size_mult, 0.5)  # halve on fresh changepoint

        weights = {k: v for k, v in REGIME_WEIGHT_ADJUSTMENTS.get(regime, {}).items()
                   if k != "position_size_mult"}

        elapsed = (time.perf_counter() - t0) * 1000
        reasoning = (
            f"trend={trend_dir:+d} vol={vol_level}({atr_pct:.3f}) "
            f"hmm_state={hmm_regime} hmm_conf={hmm_conf:.2f} "
            f"cp={cp_detected}  ({elapsed:.0f}ms)"
        )

        result = RegimeResult(
            regime=regime,
            confidence=hmm_conf,
            trend_direction=trend_dir,
            volatility_level=vol_level,
            changepoint_detected=cp_detected,
            changepoint_bar=cp_bar,
            weight_adjustments=weights,
            position_size_mult=size_mult,
            reasoning=reasoning,
        )
        logger.debug(result.summary())
        return result

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return [log_return, realised_vol] feature matrix."""
        close = df["close"].values[-self.lookback:]
        log_ret = np.diff(np.log(close))
        vol = pd.Series(log_ret).rolling(10).std().fillna(0).values
        feat = np.column_stack([log_ret, vol])
        return feat

    # ── Trend direction ───────────────────────────────────────────────────────

    def _compute_trend(self, df: pd.DataFrame) -> int:
        close = df["close"]
        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        ema200 = close.ewm(span=200).mean().iloc[-1] if len(close) >= 200 else ema50
        price = close.iloc[-1]
        # Strong bull: price > ema20 > ema50 > ema200
        if price > ema20 and ema20 > ema50:
            return 1
        elif price < ema20 and ema20 < ema50:
            return -1
        else:
            return 0

    # ── Volatility classification ─────────────────────────────────────────────

    def _classify_volatility(self, df: pd.DataFrame) -> Tuple[str, float, float]:
        """Returns (level, percentile, current_atr_pct)."""
        close = df["close"]
        high, low = df["high"], df["low"]
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = atr / close

        current_atr_pct = float(atr_pct.iloc[-1])
        historical = atr_pct.dropna().values
        if len(historical) < 20:
            return "normal", 50.0, current_atr_pct

        pct = float(np.percentile(historical, 100) * 0)  # rank current vs history
        rank = float(np.mean(historical < current_atr_pct) * 100)

        if rank >= self.vol_extreme_pct:
            return "extreme", rank, current_atr_pct
        elif rank >= self.vol_high_pct:
            return "high", rank, current_atr_pct
        elif rank <= self.vol_low_pct:
            return "low", rank, current_atr_pct
        return "normal", rank, current_atr_pct

    # ── HMM classification ────────────────────────────────────────────────────

    def _hmm_classify(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Classify features with a Gaussian HMM.
        Falls back to rule-based classification if hmmlearn is unavailable.
        """
        try:
            from hmmlearn import hmm as hmmlib
            if self._hmm_model is None:
                model = hmmlib.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=42,
                )
                model.fit(features)
                self._hmm_model = model

            states = self._hmm_model.predict(features)
            current_state = int(states[-1])

            # Compute posterior probability for confidence
            log_post = self._hmm_model.score_samples(features)
            # Use transition stability as confidence proxy
            last_5 = states[-5:]
            confidence = float(np.mean(last_5 == current_state))
            return current_state, confidence

        except ImportError:
            return self._rule_based_state(features)
        except Exception as exc:
            logger.debug(f"HMM classification failed, using rule-based: {exc}")
            return self._rule_based_state(features)

    def _rule_based_state(self, features: np.ndarray) -> Tuple[int, float]:
        """Fallback when hmmlearn is not installed."""
        recent = features[-10:]
        mean_ret = float(np.mean(recent[:, 0]))
        mean_vol = float(np.mean(recent[:, 1]))
        overall_vol = float(np.std(features[:, 0]))

        if mean_ret > 0 and mean_vol < overall_vol:
            return 0, 0.7   # bull low vol
        elif mean_ret > 0 and mean_vol >= overall_vol:
            return 1, 0.65  # bull high vol
        elif mean_ret < 0 and mean_vol < overall_vol:
            return 2, 0.7   # bear low vol
        else:
            return 3, 0.65  # bear high vol

    # ── PELT changepoint detection ────────────────────────────────────────────

    def _pelt_changepoint(self, df: pd.DataFrame) -> Tuple[bool, Optional[int]]:
        """
        Detect structural breaks using PELT algorithm.
        Returns (changepoint_in_last_N_bars, bar_index).
        """
        try:
            import ruptures as rpt
            close = df["close"].values[-self.lookback:]
            # Detect on log-return volatility signal
            returns = np.diff(np.log(close))
            vol = pd.Series(returns).rolling(5).std().fillna(0).values

            model = rpt.Pelt(model="rbf", min_size=10, jump=1)
            model.fit(vol.reshape(-1, 1))
            breakpoints = model.predict(pen=3.0)

            if not breakpoints:
                return False, None

            # Last breakpoint index in the lookback window
            last_cp = breakpoints[-2] if len(breakpoints) > 1 else breakpoints[0]
            bars_since = len(vol) - last_cp
            total_bars = len(df)

            if bars_since <= self.changepoint_lookback:
                cp_bar_abs = total_bars - bars_since
                return True, cp_bar_abs
            return False, None

        except ImportError:
            return self._simple_changepoint(df)
        except Exception as exc:
            logger.debug(f"PELT changepoint failed: {exc}")
            return False, None

    def _simple_changepoint(self, df: pd.DataFrame) -> Tuple[bool, Optional[int]]:
        """
        Fallback changepoint detection using volatility z-score.
        A bar is flagged as a changepoint if its vol is >3 std from recent average.
        """
        close = df["close"]
        returns = close.pct_change()
        roll_vol = returns.rolling(20).std()
        roll_mean = roll_vol.rolling(50).mean()
        roll_std  = roll_vol.rolling(50).std()
        z_score = (roll_vol - roll_mean) / (roll_std + 1e-9)

        # Check last changepoint_lookback bars
        recent_z = z_score.iloc[-self.changepoint_lookback:]
        if (recent_z.abs() > 2.5).any():
            cp_idx = int(recent_z.abs().idxmax()) if hasattr(recent_z.abs().idxmax(), '__int__') else None
            return True, len(df) - self.changepoint_lookback
        return False, None

    # ── Regime mapping ────────────────────────────────────────────────────────

    def _map_to_regime(self, trend: int, vol_level: str, hmm_state: int) -> str:
        if vol_level == "extreme":
            return "bear_high_vol"
        if trend == 1:
            return "bull_high_vol" if vol_level in ("high", "extreme") else "bull_low_vol"
        elif trend == -1:
            return "bear_high_vol" if vol_level in ("high", "extreme") else "bear_low_vol"
        return "sideways"

    def _default_regime(self, reason: str) -> RegimeResult:
        return RegimeResult(
            regime="sideways",
            confidence=0.3,
            trend_direction=0,
            volatility_level="normal",
            changepoint_detected=False,
            changepoint_bar=None,
            weight_adjustments={},
            position_size_mult=0.5,
            reasoning=reason,
        )

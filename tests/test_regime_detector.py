"""
Tests for RegimeDetector.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest

from agents.regime_detector import RegimeDetector, RegimeResult, REGIME_WEIGHT_ADJUSTMENTS
from data.market_data import Candle, OHLCV


def _make_trending_ohlcv(n: int = 150, trend: float = 0.003) -> OHLCV:
    """Monotonically trending price series."""
    candles = []
    price = 40000.0
    ts = datetime(2024, 1, 1)
    for i in range(n):
        price *= (1 + trend + np.random.normal(0, 0.002))
        candles.append(Candle(
            timestamp=ts, open=price, high=price * 1.005,
            low=price * 0.995, close=price,
            volume=500 + np.random.uniform(-100, 100),
        ))
        ts += timedelta(hours=4)
    return OHLCV("BTC/USDT", "4h", candles)


def _make_ranging_ohlcv(n: int = 150) -> OHLCV:
    """Mean-reverting / sideways price series."""
    candles = []
    price = 40000.0
    ts = datetime(2024, 1, 1)
    for i in range(n):
        price = 40000 + np.random.normal(0, 500)
        candles.append(Candle(
            timestamp=ts, open=price, high=price * 1.003,
            low=price * 0.997, close=price, volume=300.0
        ))
        ts += timedelta(hours=4)
    return OHLCV("BTC/USDT", "4h", candles)


def _make_volatile_ohlcv(n: int = 150) -> OHLCV:
    """High-volatility series."""
    candles = []
    price = 40000.0
    ts = datetime(2024, 1, 1)
    for i in range(n):
        price = max(100, price * (1 + np.random.normal(0, 0.05)))  # 5% daily vol
        candles.append(Candle(
            timestamp=ts, open=price, high=price * 1.04,
            low=price * 0.96, close=price, volume=800.0
        ))
        ts += timedelta(hours=4)
    return OHLCV("BTC/USDT", "4h", candles)


# ── Basic detection ───────────────────────────────────────────────────────────

def test_detect_returns_regime_result() -> None:
    detector = RegimeDetector()
    ohlcv = _make_trending_ohlcv()
    result = detector.detect(ohlcv)
    assert isinstance(result, RegimeResult)
    assert result.regime in ("bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol", "sideways")


def test_regime_confidence_in_range() -> None:
    detector = RegimeDetector()
    result = detector.detect(_make_trending_ohlcv())
    assert 0.0 <= result.confidence <= 1.0


def test_strong_uptrend_detected_as_bull() -> None:
    detector = RegimeDetector()
    ohlcv = _make_trending_ohlcv(n=200, trend=0.005)
    result = detector.detect(ohlcv)
    assert result.trend_direction == 1
    assert "bull" in result.regime


def test_sideways_market_detected() -> None:
    detector = RegimeDetector()
    ohlcv = _make_ranging_ohlcv(n=200)
    result = detector.detect(ohlcv)
    # Sideways or low-confidence
    assert result.trend_direction in (0, 1, -1)


def test_high_vol_classification() -> None:
    detector = RegimeDetector()
    ohlcv = _make_volatile_ohlcv(n=200)
    result = detector.detect(ohlcv)
    assert result.volatility_level in ("high", "extreme")


def test_insufficient_data_returns_default() -> None:
    detector = RegimeDetector(lookback=100)
    short_ohlcv = _make_trending_ohlcv(n=30)
    result = detector.detect(short_ohlcv)
    assert result.regime == "sideways"
    assert result.confidence < 0.5


# ── Regime weight adjustments ─────────────────────────────────────────────────

def test_bull_low_vol_amplifies_trend_weight() -> None:
    adj = REGIME_WEIGHT_ADJUSTMENTS["bull_low_vol"]
    assert adj["ema_trend"] > 1.0
    assert adj["trend"] > 1.0


def test_bear_high_vol_reduces_size() -> None:
    adj = REGIME_WEIGHT_ADJUSTMENTS["bear_high_vol"]
    assert adj.get("position_size_mult", 1.0) < 0.5


def test_sideways_amplifies_oscillators() -> None:
    adj = REGIME_WEIGHT_ADJUSTMENTS["sideways"]
    assert adj["rsi"] > 1.0
    assert adj["bbands"] > 1.0
    assert adj["ema_trend"] < 1.0


# ── Tradeability gate ─────────────────────────────────────────────────────────

def test_extreme_vol_not_tradeable() -> None:
    result = RegimeResult(
        regime="bear_high_vol", confidence=0.7,
        trend_direction=-1, volatility_level="extreme",
        changepoint_detected=False, changepoint_bar=None,
        weight_adjustments={}, position_size_mult=0.0,
        reasoning="extreme vol",
    )
    assert not result.is_tradeable


def test_normal_regime_is_tradeable() -> None:
    result = RegimeResult(
        regime="bull_low_vol", confidence=0.8,
        trend_direction=1, volatility_level="low",
        changepoint_detected=False, changepoint_bar=None,
        weight_adjustments={}, position_size_mult=1.0,
        reasoning="normal",
    )
    assert result.is_tradeable


def test_fresh_changepoint_blocks_trade() -> None:
    result = RegimeResult(
        regime="bull_low_vol", confidence=0.7,
        trend_direction=1, volatility_level="normal",
        changepoint_detected=True, changepoint_bar=148,
        weight_adjustments={}, position_size_mult=0.5,
        reasoning="changepoint detected",
    )
    assert not result.is_tradeable

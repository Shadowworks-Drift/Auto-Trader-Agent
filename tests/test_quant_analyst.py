"""
Tests for QuantAnalyst.
"""

from __future__ import annotations

import pytest

from agents.quant_analyst import QuantAnalyst, _compute_adx
from config.settings import Settings
from data.market_data import MarketSnapshot


@pytest.mark.asyncio
async def test_quant_analyst_returns_result(settings: Settings, market_snapshot: MarketSnapshot) -> None:
    analyst = QuantAnalyst(settings)
    result = await analyst.analyse(market_snapshot)
    assert result.success, f"QuantAnalyst failed: {result.error}"
    assert "score" in result.data
    assert "confidence" in result.data
    assert result.data["direction"] in ("long", "short", "none")


@pytest.mark.asyncio
async def test_quant_analyst_score_in_range(settings: Settings, market_snapshot: MarketSnapshot) -> None:
    analyst = QuantAnalyst(settings)
    result = await analyst.analyse(market_snapshot)
    assert -1.0 <= result.data["score"] <= 1.0
    assert 0.0 <= result.data["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_quant_analyst_indicators_present(settings: Settings, market_snapshot: MarketSnapshot) -> None:
    analyst = QuantAnalyst(settings)
    result = await analyst.analyse(market_snapshot)
    indicators = result.data.get("indicators", {})
    for key in ("rsi", "macd", "bb_upper", "bb_lower", "ema_20", "ema_50", "atr", "adx"):
        assert key in indicators, f"Missing indicator: {key}"


@pytest.mark.asyncio
async def test_quant_analyst_insufficient_data(settings: Settings) -> None:
    from datetime import datetime
    from data.market_data import Candle, OHLCV, MarketSnapshot
    tiny_ohlcv = OHLCV("BTC/USDT", "4h", [
        Candle(datetime.utcnow(), 40000, 40100, 39900, 40050, 500)
        for _ in range(5)
    ])
    snapshot = MarketSnapshot("BTC/USDT", datetime.utcnow(), {"4h": tiny_ohlcv})
    analyst = QuantAnalyst(settings)
    result = await analyst.analyse(snapshot)
    assert not result.success


def test_adx_returns_float() -> None:
    import pandas as pd
    import numpy as np
    n = 50
    price = pd.Series(np.cumsum(np.random.randn(n)) + 100)
    high = price + np.abs(np.random.randn(n)) * 0.5
    low = price - np.abs(np.random.randn(n)) * 0.5
    adx = _compute_adx(high, low, price)
    assert isinstance(adx, float)
    assert adx >= 0

"""
Tests for AltDataFetcher and AltDataBundle.
Uses mocked HTTP responses so tests run offline.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from data.alternative_data import (
    AltDataBundle,
    AltDataFetcher,
    FearGreedData,
    FundingRateData,
    LiquidationData,
    OpenInterestData,
)
from data.market_data import MarketSnapshot, OHLCV


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_fear_greed_normalised() -> None:
    fg = FearGreedData(value=25, label="Fear", timestamp=datetime.utcnow())
    assert fg.normalised == pytest.approx(-0.5, rel=1e-3)
    assert fg.contrarian_signal == pytest.approx(1.0)   # extreme fear = buy signal


def test_fear_greed_extreme_greed() -> None:
    fg = FearGreedData(value=85, label="Extreme Greed", timestamp=datetime.utcnow())
    assert fg.contrarian_signal == pytest.approx(-1.0)  # extreme greed = sell signal


def test_funding_rate_signal_crowded_longs() -> None:
    fr = FundingRateData(symbol="BTCUSDT", rate=0.10, next_funding_time=None, predicted_rate=None)
    assert fr.signal == pytest.approx(-1.0)   # crowded longs → bearish contrarian


def test_funding_rate_signal_crowded_shorts() -> None:
    fr = FundingRateData(symbol="BTCUSDT", rate=-0.05, next_funding_time=None, predicted_rate=None)
    assert fr.signal == pytest.approx(1.0)    # crowded shorts → bullish contrarian


def test_funding_rate_neutral() -> None:
    fr = FundingRateData(symbol="BTCUSDT", rate=0.01, next_funding_time=None, predicted_rate=None)
    assert fr.signal == pytest.approx(0.0)


def test_oi_signal_accumulation() -> None:
    oi = OpenInterestData(symbol="BTC/USDT", oi_usd=1e9, oi_change_24h_pct=8.0, timestamp=datetime.utcnow())
    assert oi.signal > 0


def test_liquidation_ratio_short_squeeze() -> None:
    liq = LiquidationData(
        symbol="BTC/USDT", liq_buy_24h_usd=300e6, liq_sell_24h_usd=50e6, timestamp=datetime.utcnow()
    )
    assert liq.liq_ratio > 3.0
    assert liq.signal == pytest.approx(1.0)


def test_liquidation_ratio_long_cascade() -> None:
    liq = LiquidationData(
        symbol="BTC/USDT", liq_buy_24h_usd=10e6, liq_sell_24h_usd=200e6, timestamp=datetime.utcnow()
    )
    assert liq.signal == pytest.approx(-1.0)


def test_alt_bundle_composite_signal() -> None:
    bundle = AltDataBundle(
        symbol="BTC/USDT",
        fetched_at=datetime.utcnow(),
        fear_greed=FearGreedData(value=15, label="Extreme Fear", timestamp=datetime.utcnow()),
        funding=FundingRateData(symbol="BTCUSDT", rate=-0.04, next_funding_time=None, predicted_rate=None),
        open_interest=OpenInterestData(symbol="BTC/USDT", oi_usd=1e9, oi_change_24h_pct=2.0, timestamp=datetime.utcnow()),
        liquidations=LiquidationData(symbol="BTC/USDT", liq_buy_24h_usd=200e6, liq_sell_24h_usd=50e6, timestamp=datetime.utcnow()),
    )
    # Extreme fear + crowded shorts + healthy OI + short squeeze → all bullish
    composite = bundle.composite_signal()
    assert composite > 0.5
    assert -1.0 <= composite <= 1.0


def test_alt_bundle_to_prompt_text_contains_data() -> None:
    bundle = AltDataBundle(
        symbol="ETH/USDT",
        fetched_at=datetime.utcnow(),
        fear_greed=FearGreedData(value=45, label="Fear", timestamp=datetime.utcnow()),
    )
    text = bundle.to_prompt_text()
    assert "Fear & Greed" in text
    assert "45/100" in text


def test_market_snapshot_attach_alt_data(btc_ohlcv: OHLCV) -> None:
    snapshot = MarketSnapshot(
        symbol="BTC/USDT",
        fetched_at=datetime.utcnow(),
        ohlcv={"4h": btc_ohlcv},
    )
    bundle = AltDataBundle(
        symbol="BTC/USDT",
        fetched_at=datetime.utcnow(),
        fear_greed=FearGreedData(value=30, label="Fear", timestamp=datetime.utcnow()),
        funding=FundingRateData(symbol="BTCUSDT", rate=0.03, next_funding_time=None, predicted_rate=None),
    )
    snapshot.attach_alt_data(bundle)
    assert snapshot.fear_greed_index == 30
    assert snapshot.funding_rate_annualised == pytest.approx(0.03)
    assert snapshot.alt_composite_signal is not None


@pytest.mark.asyncio
async def test_fetcher_graceful_failure() -> None:
    """Fetcher should return a bundle with None fields if APIs are unavailable."""
    fetcher = AltDataFetcher()
    with patch.object(fetcher, "_fetch_fear_greed", new_callable=AsyncMock, return_value=None), \
         patch.object(fetcher, "_fetch_funding_rate", new_callable=AsyncMock, return_value=None), \
         patch.object(fetcher, "_fetch_open_interest", new_callable=AsyncMock, return_value=None), \
         patch.object(fetcher, "_fetch_liquidations", new_callable=AsyncMock, return_value=None):
        bundle = await fetcher.fetch("BTC/USDT")
    assert bundle.fear_greed is None
    assert bundle.funding is None
    assert bundle.composite_signal() == pytest.approx(0.0)

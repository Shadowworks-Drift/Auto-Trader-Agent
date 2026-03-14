"""
Shared test fixtures.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pytest

from config.settings import Settings
from data.market_data import Candle, OHLCV, MarketSnapshot


def _make_candles(n: int = 200, start_price: float = 40000.0) -> List[Candle]:
    """Generate synthetic candles for testing."""
    import random
    candles = []
    price = start_price
    ts = datetime(2024, 1, 1)
    for i in range(n):
        change = random.gauss(0, 0.01)
        price = max(1.0, price * (1 + change))
        high = price * (1 + abs(random.gauss(0, 0.005)))
        low = price * (1 - abs(random.gauss(0, 0.005)))
        volume = random.uniform(100, 1000)
        candles.append(Candle(timestamp=ts, open=price, high=high, low=low, close=price, volume=volume))
        ts += timedelta(hours=4)
    return candles


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def btc_ohlcv() -> OHLCV:
    return OHLCV(symbol="BTC/USDT", timeframe="4h", candles=_make_candles(200))


@pytest.fixture
def market_snapshot(btc_ohlcv: OHLCV) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT",
        fetched_at=datetime.utcnow(),
        ohlcv={"4h": btc_ohlcv, "1d": btc_ohlcv, "1h": btc_ohlcv},
    )

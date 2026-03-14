"""
Symbol Selector — screens a universe of assets down to tradeable candidates.

Filters by:
  - 24h quote volume (liquidity floor)
  - ATR % (volatility band — too flat or too wild are excluded)
  - Top-N by composite momentum score
"""

from __future__ import annotations

import asyncio
from typing import Dict, List

import numpy as np
from loguru import logger

from config.settings import Settings
from data.data_sync import DataSync
from data.market_data import OHLCV


class SymbolScore:
    def __init__(self, symbol: str, volume: float, atr_pct: float, momentum: float) -> None:
        self.symbol = symbol
        self.volume = volume
        self.atr_pct = atr_pct
        self.momentum = momentum

    @property
    def composite(self) -> float:
        """Normalised composite score (higher = more attractive)."""
        vol_score = min(self.volume / 1e9, 1.0)          # cap at 1 B USDT
        mom_score = max(0.0, min(self.momentum, 1.0))
        atr_score = min(self.atr_pct / 0.05, 1.0)        # sweet spot ~5% ATR
        return 0.4 * vol_score + 0.4 * mom_score + 0.2 * atr_score


class SymbolSelector:
    """
    Selects the most tradeable symbols from the configured universe.

    Usage::

        selector = SymbolSelector(settings, data_sync)
        selected = await selector.select(settings.trading.symbols)
    """

    def __init__(self, settings: Settings, data_sync: DataSync) -> None:
        self.settings = settings
        self.data_sync = data_sync
        self.cfg = settings.symbol_selector

    async def select(self, universe: List[str]) -> List[str]:
        """Return ordered list of symbols passing all filters."""
        scores: List[SymbolScore] = []

        tasks = [self._score_symbol(sym) for sym in universe]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sym, result in zip(universe, results):
            if isinstance(result, SymbolScore):
                scores.append(result)
            else:
                logger.debug(f"Symbol {sym} scoring failed: {result}")

        # Apply filters
        filtered = [
            s for s in scores
            if s.volume >= self.cfg.min_volume_usdt_24h
            and self.cfg.min_atr_pct <= s.atr_pct <= self.cfg.max_atr_pct
        ]

        # Sort by composite score descending, take top-N
        filtered.sort(key=lambda s: s.composite, reverse=True)
        selected = [s.symbol for s in filtered[: self.cfg.top_n]]

        logger.info(
            f"SymbolSelector: {len(universe)} → {len(filtered)} passed filters "
            f"→ {len(selected)} selected: {selected}"
        )
        return selected

    async def _score_symbol(self, symbol: str) -> SymbolScore:
        primary_tf = self.settings.trading.timeframes["primary"]
        ohlcv = (await self.data_sync.fetch_snapshot(symbol)).ohlcv.get(primary_tf)
        volume_24h = await self.data_sync.fetch_ticker_volume(symbol)

        if ohlcv is None or len(ohlcv) < 20:
            raise ValueError(f"Insufficient data for {symbol}")

        atr_pct = _compute_atr_pct(ohlcv)
        momentum = _compute_momentum(ohlcv)

        return SymbolScore(
            symbol=symbol,
            volume=volume_24h,
            atr_pct=atr_pct,
            momentum=momentum,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_atr_pct(ohlcv: OHLCV, period: int = 14) -> float:
    """ATR as a fraction of current price."""
    df = ohlcv.df.tail(period + 1)
    if len(df) < 2:
        return 0.0
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            np.abs(df["high"] - df["close"].shift(1)),
            np.abs(df["low"] - df["close"].shift(1)),
        ),
    )
    atr = float(tr.rolling(period).mean().iloc[-1])
    price = float(df["close"].iloc[-1])
    return atr / price if price > 0 else 0.0


def _compute_momentum(ohlcv: OHLCV, period: int = 14) -> float:
    """Rate of change momentum, normalised to [0, 1]."""
    closes = ohlcv.df["close"].values
    if len(closes) < period + 1:
        return 0.5
    roc = (closes[-1] - closes[-period - 1]) / closes[-period - 1]
    return float(min(max((roc + 0.3) / 0.6, 0.0), 1.0))

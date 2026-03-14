"""
DataSync — fetches and caches multi-timeframe OHLCV data using ccxt.
Supports async concurrent fetching across symbols and timeframes.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import ccxt.async_support as ccxt
from loguru import logger

from config.settings import Settings
from .market_data import Candle, MarketSnapshot, NewsItem, OHLCV


class OHLCVCache:
    """Simple TTL cache for OHLCV data."""

    def __init__(self, ttl_seconds: int = 60) -> None:
        self._store: Dict[str, Tuple[float, OHLCV]] = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[OHLCV]:
        if key in self._store:
            ts, data = self._store[key]
            if time.time() - ts < self.ttl:
                return data
        return None

    def set(self, key: str, data: OHLCV) -> None:
        self._store[key] = (time.time(), data)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)


class DataSync:
    """
    Multi-timeframe data synchroniser.

    Fetches OHLCV candles concurrently from a CCXT exchange and bundles them
    into MarketSnapshot objects consumed by the agent pipeline.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = OHLCVCache(ttl_seconds=settings.data.cache_ttl_seconds)
        self._exchange: Optional[ccxt.Exchange] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        cfg = self.settings.exchange
        exchange_class = getattr(ccxt, cfg.id)
        params: Dict = {
            "enableRateLimit": True,
            "rateLimit": cfg.rate_limit_ms,
        }
        if cfg.api_key:
            params["apiKey"] = cfg.api_key
        if cfg.api_secret:
            params["secret"] = cfg.api_secret
        if cfg.sandbox:
            params["sandbox"] = True

        self._exchange = exchange_class(params)
        logger.info(f"DataSync connected to {cfg.id} (sandbox={cfg.sandbox})")

    async def disconnect(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_snapshot(self, symbol: str) -> MarketSnapshot:
        """Return a complete MarketSnapshot for a single symbol."""
        tfs = self.settings.trading.timeframes
        timeframe_list = list(dict.fromkeys([tfs["context"], tfs["primary"], tfs["entry"]]))

        tasks = [self._fetch_ohlcv(symbol, tf) for tf in timeframe_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ohlcv_map: Dict[str, OHLCV] = {}
        for tf, result in zip(timeframe_list, results):
            if isinstance(result, OHLCV):
                ohlcv_map[tf] = result
            else:
                logger.warning(f"Failed to fetch {symbol} {tf}: {result}")

        news: List[NewsItem] = []
        if self.settings.data.news_enabled:
            news = await self._fetch_news(symbol)

        return MarketSnapshot(
            symbol=symbol,
            fetched_at=datetime.utcnow(),
            ohlcv=ohlcv_map,
            news=news,
        )

    async def fetch_all_snapshots(self, symbols: List[str]) -> Dict[str, MarketSnapshot]:
        """Concurrently fetch snapshots for all symbols."""
        tasks = [self.fetch_snapshot(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        snapshots: Dict[str, MarketSnapshot] = {}
        for sym, result in zip(symbols, results):
            if isinstance(result, MarketSnapshot):
                snapshots[sym] = result
            else:
                logger.error(f"Snapshot failed for {sym}: {result}")
        return snapshots

    async def fetch_ticker_volume(self, symbol: str) -> float:
        """Fetch 24h volume in quote currency (USDT) for symbol screening."""
        if not self._exchange:
            return 0.0
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            return float(ticker.get("quoteVolume") or ticker.get("baseVolume") or 0)
        except Exception as exc:
            logger.debug(f"Ticker fetch failed for {symbol}: {exc}")
            return 0.0

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _fetch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        cache_key = f"{symbol}:{timeframe}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        if not self._exchange:
            raise RuntimeError("DataSync not connected — call connect() first")

        limit = self.settings.data.ohlcv_limit
        raw = await self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        candles = [Candle.from_ccxt(row) for row in raw]
        result = OHLCV(symbol=symbol, timeframe=timeframe, candles=candles)
        self.cache.set(cache_key, result)
        logger.debug(f"Fetched {len(candles)} candles for {symbol} {timeframe}")
        return result

    async def _fetch_news(self, symbol: str) -> List[NewsItem]:
        """Placeholder — integrates with CryptoPanic API when key is configured."""
        api_key = self.settings.data.news_api_key
        if not api_key:
            return []
        try:
            import httpx
            base_currency = symbol.split("/")[0]
            url = (
                f"https://cryptopanic.com/api/v1/posts/"
                f"?auth_token={api_key}&currencies={base_currency}&public=true"
            )
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            items: List[NewsItem] = []
            for post in data.get("results", [])[:20]:
                items.append(
                    NewsItem(
                        timestamp=datetime.fromisoformat(
                            post["published_at"].replace("Z", "+00:00")
                        ),
                        title=post["title"],
                        source=post.get("source", {}).get("title", "unknown"),
                        url=post.get("url", ""),
                    )
                )
            return items
        except Exception as exc:
            logger.warning(f"News fetch failed for {symbol}: {exc}")
            return []

"""
Alternative Data Fetcher
════════════════════════
Pulls free-tier crypto alternative data from:

  • Alternative.me      — Fear & Greed Index
  • CoinGlass           — Funding rates, Open Interest, Liquidation levels
  • CoinGecko           — Exchange net-flow proxy, volume breakdown
  • Blockchain.info     — BTC mempool / on-chain basics (no key required)

All fetchers are async, cache results with a short TTL, and fail gracefully
(returning None fields rather than raising) so the main pipeline keeps running
even when external services are down.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import httpx
from loguru import logger


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class FearGreedData:
    value: int                    # 0 (extreme fear) – 100 (extreme greed)
    label: str                    # "Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"
    timestamp: datetime
    prev_day_value: Optional[int] = None
    prev_week_value: Optional[int] = None

    @property
    def normalised(self) -> float:
        """−1 (extreme fear) to +1 (extreme greed)."""
        return (self.value - 50) / 50.0

    @property
    def contrarian_signal(self) -> float:
        """Contrarian: extreme readings are mean-reversion signals.
        Returns +1 (buy signal) when extreme fear, −1 when extreme greed."""
        if self.value <= 20:
            return 1.0
        elif self.value >= 80:
            return -1.0
        return 0.0


@dataclass
class FundingRateData:
    symbol: str
    rate: float                   # annualised %
    next_funding_time: Optional[datetime]
    predicted_rate: Optional[float]

    @property
    def signal(self) -> float:
        """Positive funding = crowded longs = bearish contrarian.
        Returns −1 when very positive (crowded), +1 when very negative (crowded shorts)."""
        if self.rate > 0.05:      # > 5% annualised → crowded longs
            return -1.0
        elif self.rate < -0.02:   # < −2% annualised → crowded shorts
            return 1.0
        return 0.0


@dataclass
class OpenInterestData:
    symbol: str
    oi_usd: float
    oi_change_24h_pct: float      # % change
    timestamp: datetime

    @property
    def signal(self) -> float:
        """OI rising + price rising = new longs = trend confirmation (+1).
        OI rising + price falling = new shorts = trend confirmation of down move.
        Returned as ±0.5 — contextual, not standalone."""
        if self.oi_change_24h_pct > 5:
            return 0.5   # accumulation
        elif self.oi_change_24h_pct < -10:
            return -0.5  # deleverage / fear
        return 0.0


@dataclass
class LiquidationData:
    symbol: str
    liq_buy_24h_usd: float        # short liquidations (price went up)
    liq_sell_24h_usd: float       # long liquidations (price went down)
    timestamp: datetime

    @property
    def liq_ratio(self) -> float:
        """Ratio of buy/sell liquidations. > 1 means more shorts being squeezed."""
        denom = max(self.liq_sell_24h_usd, 1.0)
        return self.liq_buy_24h_usd / denom

    @property
    def signal(self) -> float:
        r = self.liq_ratio
        if r > 3.0:
            return 1.0    # heavy short liquidations → bullish momentum
        elif r < 0.33:
            return -1.0   # heavy long liquidations → bearish momentum
        return 0.0


@dataclass
class AltDataBundle:
    """All alternative data for one symbol at one point in time."""
    symbol: str
    fetched_at: datetime
    fear_greed: Optional[FearGreedData] = None
    funding: Optional[FundingRateData] = None
    open_interest: Optional[OpenInterestData] = None
    liquidations: Optional[LiquidationData] = None
    extra: Dict[str, float] = field(default_factory=dict)

    def composite_signal(self) -> float:
        """Weighted composite alternative data signal in [−1, +1]."""
        components: list[tuple[float, float]] = []  # (signal, weight)
        if self.fear_greed is not None:
            components.append((self.fear_greed.contrarian_signal, 0.25))
        if self.funding is not None:
            components.append((self.funding.signal, 0.30))
        if self.open_interest is not None:
            components.append((self.open_interest.signal, 0.20))
        if self.liquidations is not None:
            components.append((self.liquidations.signal, 0.25))
        if not components:
            return 0.0
        total_w = sum(w for _, w in components)
        return sum(s * w for s, w in components) / total_w

    def to_prompt_text(self) -> str:
        """Human-readable summary for LLM prompts."""
        lines: list[str] = [f"Alternative data for {self.symbol}:"]
        if self.fear_greed:
            fg = self.fear_greed
            lines.append(
                f"  Fear & Greed: {fg.value}/100 ({fg.label})  "
                f"prev_day={fg.prev_day_value}  prev_week={fg.prev_week_value}"
            )
        if self.funding:
            f = self.funding
            lines.append(
                f"  Funding Rate: {f.rate:+.4f}%/annualised  "
                f"(signal: {'crowded longs' if f.rate > 0.03 else 'crowded shorts' if f.rate < -0.01 else 'neutral'})"
            )
        if self.open_interest:
            oi = self.open_interest
            lines.append(
                f"  Open Interest: ${oi.oi_usd:,.0f}  change_24h={oi.oi_change_24h_pct:+.1f}%"
            )
        if self.liquidations:
            liq = self.liquidations
            lines.append(
                f"  Liquidations 24h: buy=${liq.liq_buy_24h_usd:,.0f} "
                f"sell=${liq.liq_sell_24h_usd:,.0f}  ratio={liq.liq_ratio:.2f}"
            )
        if self.extra:
            for k, v in self.extra.items():
                lines.append(f"  {k}: {v}")
        lines.append(f"  Composite signal: {self.composite_signal():+.3f}")
        return "\n".join(lines)


# ── TTL Cache ─────────────────────────────────────────────────────────────────

class _TTLCache:
    def __init__(self, ttl: int = 300) -> None:
        self._store: dict = {}
        self.ttl = ttl

    def get(self, key: str):
        if key in self._store:
            ts, val = self._store[key]
            if time.time() - ts < self.ttl:
                return val
        return None

    def set(self, key: str, val) -> None:
        self._store[key] = (time.time(), val)


# ── Fetcher ───────────────────────────────────────────────────────────────────

class AltDataFetcher:
    """
    Fetches alternative data from free public APIs.

    Usage::

        fetcher = AltDataFetcher()
        bundle = await fetcher.fetch("BTC/USDT")
        print(bundle.to_prompt_text())
        print(f"Composite signal: {bundle.composite_signal()}")
    """

    FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=7&format=json"
    COINGLASS_BASE  = "https://open-api.coinglass.com/public/v2"
    COINGECKO_BASE  = "https://api.coingecko.com/api/v3"

    def __init__(self, coinglass_key: str = "", timeout: float = 10.0) -> None:
        self.coinglass_key = coinglass_key
        self.timeout = timeout
        self._cache = _TTLCache(ttl=300)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "AltDataFetcher":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    # ── Public ────────────────────────────────────────────────────────────────

    async def fetch(self, symbol: str) -> AltDataBundle:
        """Fetch all alternative data for a symbol concurrently."""
        base = symbol.split("/")[0].upper()           # "BTC/USDT" → "BTC"
        perp = f"{base}USDT"                          # for CoinGlass / exchange notation

        fg_task   = self._fetch_fear_greed()
        fund_task = self._fetch_funding_rate(perp)
        oi_task   = self._fetch_open_interest(perp, symbol)
        liq_task  = self._fetch_liquidations(perp, symbol)

        fg, fund, oi, liq = await asyncio.gather(
            fg_task, fund_task, oi_task, liq_task, return_exceptions=True
        )

        return AltDataBundle(
            symbol=symbol,
            fetched_at=datetime.utcnow(),
            fear_greed=fg if isinstance(fg, FearGreedData) else None,
            funding=fund if isinstance(fund, FundingRateData) else None,
            open_interest=oi if isinstance(oi, OpenInterestData) else None,
            liquidations=liq if isinstance(liq, LiquidationData) else None,
        )

    # ── Fear & Greed ──────────────────────────────────────────────────────────

    async def _fetch_fear_greed(self) -> Optional[FearGreedData]:
        key = "fear_greed"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            resp = await self._http().get(self.FEAR_GREED_URL)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                return None
            current = data[0]
            result = FearGreedData(
                value=int(current["value"]),
                label=current["value_classification"],
                timestamp=datetime.utcfromtimestamp(int(current["timestamp"])),
                prev_day_value=int(data[1]["value"]) if len(data) > 1 else None,
                prev_week_value=int(data[6]["value"]) if len(data) > 6 else None,
            )
            self._cache.set(key, result)
            return result
        except Exception as exc:
            logger.debug(f"Fear & Greed fetch failed: {exc}")
            return None

    # ── Funding Rate ──────────────────────────────────────────────────────────

    async def _fetch_funding_rate(self, perp: str) -> Optional[FundingRateData]:
        key = f"funding:{perp}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            # CoinGlass open API (no key required for basic endpoints)
            url = f"{self.COINGLASS_BASE}/funding_rate_history"
            headers = {}
            if self.coinglass_key:
                headers["coinglassSecret"] = self.coinglass_key
            resp = await self._http().get(
                url, params={"symbol": perp, "interval": 8, "limit": 1},
                headers=headers
            )
            if resp.status_code == 200:
                rows = resp.json().get("data", [])
                if rows:
                    latest = rows[-1]
                    rate = float(latest.get("fundingRate", 0)) * 3 * 365 * 100  # → annualised %
                    result = FundingRateData(
                        symbol=perp,
                        rate=rate,
                        next_funding_time=None,
                        predicted_rate=None,
                    )
                    self._cache.set(key, result)
                    return result
            # Fallback: CoinGecko derivatives endpoint
            return await self._fetch_funding_coingecko(perp)
        except Exception as exc:
            logger.debug(f"Funding rate fetch failed for {perp}: {exc}")
            return None

    async def _fetch_funding_coingecko(self, perp: str) -> Optional[FundingRateData]:
        """CoinGecko derivatives fallback — less precise but no key required."""
        try:
            base = perp.replace("USDT", "").lower()
            resp = await self._http().get(
                f"{self.COINGECKO_BASE}/derivatives",
                params={"include_tickers": "unexpired"}
            )
            resp.raise_for_status()
            tickers = resp.json()
            matching = [
                t for t in tickers
                if t.get("base", "").upper() == base.upper()
                and "perp" in t.get("contract_type", "").lower()
            ]
            if matching:
                rates = [float(t.get("funding_rate", 0)) for t in matching[:5] if t.get("funding_rate")]
                if rates:
                    avg_rate = sum(rates) / len(rates) * 3 * 365 * 100
                    return FundingRateData(symbol=perp, rate=avg_rate, next_funding_time=None, predicted_rate=None)
        except Exception as exc:
            logger.debug(f"CoinGecko funding fallback failed: {exc}")
        return None

    # ── Open Interest ─────────────────────────────────────────────────────────

    async def _fetch_open_interest(self, perp: str, symbol: str) -> Optional[OpenInterestData]:
        key = f"oi:{perp}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            headers = {}
            if self.coinglass_key:
                headers["coinglassSecret"] = self.coinglass_key
            resp = await self._http().get(
                f"{self.COINGLASS_BASE}/open_interest",
                params={"symbol": perp},
                headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                oi_usd = float(data.get("oiUsd", 0))
                change_pct = float(data.get("h24Change", 0))
                result = OpenInterestData(
                    symbol=symbol,
                    oi_usd=oi_usd,
                    oi_change_24h_pct=change_pct,
                    timestamp=datetime.utcnow(),
                )
                self._cache.set(key, result)
                return result
        except Exception as exc:
            logger.debug(f"Open interest fetch failed for {perp}: {exc}")
        return None

    # ── Liquidations ──────────────────────────────────────────────────────────

    async def _fetch_liquidations(self, perp: str, symbol: str) -> Optional[LiquidationData]:
        key = f"liq:{perp}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            headers = {}
            if self.coinglass_key:
                headers["coinglassSecret"] = self.coinglass_key
            resp = await self._http().get(
                f"{self.COINGLASS_BASE}/liquidation_history",
                params={"symbol": perp, "interval": "24h"},
                headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                result = LiquidationData(
                    symbol=symbol,
                    liq_buy_24h_usd=float(data.get("buyUsd", 0)),
                    liq_sell_24h_usd=float(data.get("sellUsd", 0)),
                    timestamp=datetime.utcnow(),
                )
                self._cache.set(key, result)
                return result
        except Exception as exc:
            logger.debug(f"Liquidations fetch failed for {perp}: {exc}")
        return None

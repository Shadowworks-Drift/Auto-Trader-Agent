"""
Domain models for market data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from data.alternative_data import AltDataBundle


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_ccxt(cls, row: list) -> "Candle":
        """Parse a raw CCXT OHLCV row [ts_ms, o, h, l, c, v]."""
        return cls(
            timestamp=datetime.utcfromtimestamp(row[0] / 1000),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )


class OHLCV:
    """A labelled, pandas-backed OHLCV series with helper methods."""

    def __init__(self, symbol: str, timeframe: str, candles: List[Candle]) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.candles = candles
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.DataFrame(
                [
                    {
                        "timestamp": c.timestamp,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                    }
                    for c in self.candles
                ]
            ).set_index("timestamp")
        return self._df

    @property
    def current_price(self) -> float:
        return self.candles[-1].close if self.candles else 0.0

    @property
    def current_volume(self) -> float:
        return self.candles[-1].volume if self.candles else 0.0

    def summary(self, n: int = 5) -> str:
        """Human-readable recent candle summary for LLM prompts."""
        recent = self.candles[-n:]
        lines = [f"{self.symbol} ({self.timeframe}) — last {len(recent)} candles:"]
        for c in recent:
            lines.append(
                f"  {c.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                f"O={c.open:.4f}  H={c.high:.4f}  L={c.low:.4f}  C={c.close:.4f}  V={c.volume:.1f}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.candles)


@dataclass
class NewsItem:
    timestamp: datetime
    title: str
    source: str
    url: str = ""
    sentiment_score: float = 0.0   # pre-computed if available


@dataclass
class MarketSnapshot:
    """Aggregated view of a symbol at a single point in time."""
    symbol: str
    fetched_at: datetime
    ohlcv: Dict[str, OHLCV]          # keyed by timeframe
    news: List[NewsItem] = field(default_factory=list)
    social_signals: Dict[str, float] = field(default_factory=dict)  # e.g. fear_greed
    on_chain: Dict[str, float] = field(default_factory=dict)

    # ── Alternative data (populated when alt-data branch is active) ───────────
    alt_data: Optional["AltDataBundle"] = None

    # Convenience numeric fields promoted from alt_data for fast access
    fear_greed_index: Optional[int] = None          # 0–100
    funding_rate_annualised: Optional[float] = None # % annualised
    oi_change_24h_pct: Optional[float] = None       # %
    liq_ratio: Optional[float] = None               # buy_liq / sell_liq
    alt_composite_signal: Optional[float] = None    # −1 to +1 composite

    @property
    def current_price(self) -> float:
        for tf in ("1h", "4h", "1d"):
            if tf in self.ohlcv:
                return self.ohlcv[tf].current_price
        return 0.0

    def attach_alt_data(self, bundle: "AltDataBundle") -> None:
        """Attach an AltDataBundle and promote key numeric fields."""
        self.alt_data = bundle
        if bundle.fear_greed:
            self.fear_greed_index = bundle.fear_greed.value
        if bundle.funding:
            self.funding_rate_annualised = bundle.funding.rate
        if bundle.open_interest:
            self.oi_change_24h_pct = bundle.open_interest.oi_change_24h_pct
        if bundle.liquidations:
            self.liq_ratio = bundle.liquidations.liq_ratio
        self.alt_composite_signal = bundle.composite_signal()

    def news_headlines_text(self, max_items: int = 10) -> str:
        items = self.news[:max_items]
        if not items:
            return "No recent news available."
        return "\n".join(
            f"- [{n.source}] {n.title} ({n.timestamp.strftime('%Y-%m-%d %H:%M')})"
            for n in items
        )

    def alt_data_text(self) -> str:
        if self.alt_data:
            return self.alt_data.to_prompt_text()
        parts = []
        if self.fear_greed_index is not None:
            parts.append(f"  Fear & Greed: {self.fear_greed_index}/100")
        if self.funding_rate_annualised is not None:
            parts.append(f"  Funding Rate: {self.funding_rate_annualised:+.3f}% annualised")
        if self.oi_change_24h_pct is not None:
            parts.append(f"  OI Change 24h: {self.oi_change_24h_pct:+.1f}%")
        if self.liq_ratio is not None:
            parts.append(f"  Liquidation Ratio (buy/sell): {self.liq_ratio:.2f}")
        return "\n".join(parts) if parts else "No alternative data available."

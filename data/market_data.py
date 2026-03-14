"""
Domain models for market data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


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

    @property
    def current_price(self) -> float:
        for tf in ("1h", "4h", "1d"):
            if tf in self.ohlcv:
                return self.ohlcv[tf].current_price
        return 0.0

    def news_headlines_text(self, max_items: int = 10) -> str:
        items = self.news[:max_items]
        if not items:
            return "No recent news available."
        return "\n".join(
            f"- [{n.source}] {n.title} ({n.timestamp.strftime('%Y-%m-%d %H:%M')})"
            for n in items
        )

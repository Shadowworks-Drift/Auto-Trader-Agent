"""
Chart Renderer — converts an OHLCV series into a PNG image in memory.

Used by ChartVisionAgent to produce a base64-encoded chart that is sent
to a local Ollama vision model (llava, moondream, etc.) for visual
pattern recognition.

No disk I/O — everything stays in memory as bytes.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

from loguru import logger

from data.market_data import OHLCV


def render_chart(
    ohlcv: OHLCV,
    n_candles: int = 80,
    figsize: tuple = (12, 7),
    dpi: int = 100,
) -> Optional[bytes]:
    """
    Render a candlestick chart with EMA 20/50 overlay and volume subplot.

    Parameters
    ----------
    ohlcv:      OHLCV instance (must have at least 20 candles)
    n_candles:  Number of most-recent candles to display
    figsize:    Matplotlib figure size (width, height) in inches
    dpi:        Dots-per-inch for the output PNG

    Returns
    -------
    Raw PNG bytes, or None if rendering fails (missing deps, bad data).
    """
    try:
        import mplfinance as mpf
        import pandas as pd
    except ImportError:
        logger.warning("chart_renderer: mplfinance not installed — run: pip install mplfinance Pillow")
        return None

    df = ohlcv.df.copy()
    if len(df) < 20:
        logger.debug(f"chart_renderer: not enough candles ({len(df)}) for {ohlcv.symbol}")
        return None

    # Slice to the most-recent n_candles
    df = df.tail(n_candles)

    # mplfinance expects columns: Open, High, Low, Close, Volume (title-case)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})

    close = df["Close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    apds = [
        mpf.make_addplot(ema20, color="#f0a500", label="EMA20"),
        mpf.make_addplot(ema50, color="#3a9bd5", label="EMA50"),
    ]

    buf = io.BytesIO()
    mpf.plot(
        df,
        type="candle",
        style="nightclouds",
        addplot=apds,
        volume=True,
        title=f"{ohlcv.symbol} · {ohlcv.timeframe} · last {len(df)} candles",
        figsize=figsize,
        savefig=dict(fname=buf, dpi=dpi, bbox_inches="tight"),
    )
    buf.seek(0)
    return buf.read()


def chart_to_base64(png_bytes: bytes) -> str:
    """Encode raw PNG bytes to a base64 string suitable for Ollama's image field."""
    return base64.b64encode(png_bytes).decode("utf-8")

"""
ChartVisionAgent — visual pattern recognition via a local Ollama vision model.

Renders the primary-timeframe candlestick chart to a PNG, encodes it as
base64, then asks a vision-capable LLM (llava, moondream, etc.) to identify:

  • Overall trend direction
  • Named chart patterns (H&S, double top/bottom, flag, wedge, triangle …)
  • Key support / resistance levels
  • Volume confirmation or divergence
  • Entry bias + confidence

The result is an AgentResult that slots into DecisionCore's weighted fusion
exactly like the other five agents.  If the vision model is unavailable or
rendering fails, the agent returns success=False and the weight drops out
of the fusion automatically — trading continues unaffected.

Prerequisites
─────────────
  pip install mplfinance Pillow          # chart rendering
  ollama pull llava:7b                   # or moondream:latest (smaller/faster)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from loguru import logger

from config.settings import Settings
from data.chart_renderer import chart_to_base64, render_chart
from data.market_data import MarketSnapshot
from llm.ollama_client import OllamaClient
from .base_agent import AgentResult, BaseAgent

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert technical analyst specialising in candlestick chart pattern recognition.
You will be shown a OHLCV candlestick chart with EMA 20 (orange) and EMA 50 (blue) overlays
plus a volume subplot.

Your job is to visually identify price action patterns, trend structure, and key levels that
numerical indicators alone may miss.

Always respond with a single valid JSON object — no prose, no markdown fences outside the JSON.
"""

# ── User prompt template ───────────────────────────────────────────────────────

_USER_TEMPLATE = """\
Symbol: {symbol}
Timeframe: {timeframe}
Current price: {current_price}
Quantitative context: {quant_summary}

Analyse the chart image and return JSON in EXACTLY this schema:
{{
  "trend": "bullish" | "bearish" | "sideways",
  "trend_strength": <1-10 integer>,
  "patterns": ["<pattern_name>", ...],
  "key_levels": {{
    "support": <float or null>,
    "resistance": <float or null>
  }},
  "volume_signal": "confirming" | "diverging" | "neutral",
  "entry_bias": "long" | "short" | "none",
  "confidence": <0.0-1.0 float>,
  "reasoning": "<one sentence visual summary>"
}}

Recognisable patterns include (non-exhaustive):
  head_and_shoulders, inverse_head_and_shoulders, double_top, double_bottom,
  ascending_triangle, descending_triangle, symmetrical_triangle,
  bull_flag, bear_flag, bull_pennant, bear_pennant,
  rising_wedge, falling_wedge, cup_and_handle,
  engulfing_bullish, engulfing_bearish, doji, hammer, shooting_star,
  morning_star, evening_star, consolidation_box, breakout, breakdown
"""


class ChartVisionAgent(BaseAgent):
    """Analyses a rendered candlestick chart using a local vision LLM."""

    name = "ChartVisionAgent"

    def __init__(self, settings: Settings, llm: OllamaClient) -> None:
        super().__init__()
        self.settings = settings
        self.llm = llm

    async def analyse(
        self,
        snapshot: MarketSnapshot,
        quant_summary: str = "",
    ) -> AgentResult:
        t0 = time.perf_counter()

        if not self.settings.llm.vision_enabled:
            return self._make_result(
                snapshot.symbol, success=False, data={},
                error="vision_enabled=false in config",
            )

        # ── Select primary timeframe ───────────────────────────────────────────
        primary_tf = self.settings.trading.timeframes.get("primary", "4h")
        ohlcv = snapshot.ohlcv.get(primary_tf)
        if ohlcv is None or len(ohlcv) < 20:
            return self._make_result(
                snapshot.symbol, success=False, data={},
                error=f"No OHLCV data for timeframe {primary_tf}",
            )

        # ── Render chart → PNG bytes → base64 ─────────────────────────────────
        png_bytes = render_chart(ohlcv, n_candles=80)
        if png_bytes is None:
            return self._make_result(
                snapshot.symbol, success=False, data={},
                error="Chart rendering failed (mplfinance unavailable or bad data)",
            )
        image_b64 = chart_to_base64(png_bytes)

        # ── Build prompt ───────────────────────────────────────────────────────
        user_msg = _USER_TEMPLATE.format(
            symbol=snapshot.symbol,
            timeframe=primary_tf,
            current_price=f"{snapshot.current_price:.4f}",
            quant_summary=quant_summary or "N/A",
        )

        # ── Call vision model ──────────────────────────────────────────────────
        try:
            resp = await self.llm.chat_vision(
                image_b64=image_b64,
                user_msg=user_msg,
                system=_SYSTEM,
                expect_json=True,
            )
        except Exception as exc:
            logger.warning(
                f"ChartVisionAgent: vision model call failed for {snapshot.symbol}: {exc}. "
                f"Is '{self.settings.llm.vision_model}' pulled? "
                f"Run: ollama pull {self.settings.llm.vision_model}"
            )
            return self._make_result(snapshot.symbol, success=False, data={}, error=str(exc))

        if not resp.parsed:
            logger.warning(
                f"ChartVisionAgent: could not parse JSON from vision response for "
                f"{snapshot.symbol} — raw: {resp.content[:400]!r}"
            )
            return self._make_result(snapshot.symbol, success=False, data={}, error="JSON parse failed")

        data = resp.parsed
        trend = data.get("trend", "sideways").lower()
        bias = data.get("entry_bias", "none").lower()
        # Derive direction: prefer explicit entry_bias, fall back to trend
        if bias in ("long", "short"):
            direction = bias
        elif trend == "bullish":
            direction = "long"
        elif trend == "bearish":
            direction = "short"
        else:
            direction = "none"
        data["direction"] = direction

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"ChartVisionAgent [{snapshot.symbol}] → {direction.upper()} "
            f"conf={data.get('confidence', 0):.2f} "
            f"patterns={data.get('patterns', [])} "
            f"({elapsed:.0f}ms)"
        )
        return self._make_result(snapshot.symbol, success=True, data=data, elapsed_ms=elapsed)

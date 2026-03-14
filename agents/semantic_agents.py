"""
Semantic Agents — LLM-powered analysis modules.

Four agents:
  1. TrendAgent      — macro trend assessment (context + primary TF)
  2. SetupAgent      — trade setup quality evaluation
  3. TriggerAgent    — precise entry trigger on lowest TF
  4. SentimentAgent  — news / social signal interpretation
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from loguru import logger

from config.settings import Settings
from data.market_data import MarketSnapshot
from llm.ollama_client import OllamaClient
from llm.prompts import PromptLibrary
from .base_agent import AgentResult, BaseAgent


class _LLMAgent(BaseAgent):
    """Shared base for LLM-backed agents."""

    def __init__(self, settings: Settings, llm: OllamaClient) -> None:
        super().__init__()
        self.settings = settings
        self.llm = llm

    async def _ask(
        self,
        system: str,
        user_msg: str,
        symbol: str,
    ) -> Dict[str, Any]:
        messages = [{"role": "user", "content": user_msg}]
        resp = await self.llm.chat(
            messages=messages,
            system=system,
            expect_json=True,
        )
        if resp.parsed:
            return resp.parsed
        logger.warning(f"{self.name}: could not parse JSON from LLM response for {symbol}")
        return {"confidence": 0.0, "reasoning": resp.content[:300]}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Trend Agent
# ══════════════════════════════════════════════════════════════════════════════

class TrendAgent(_LLMAgent):
    """Analyses the macro trend using context + primary timeframe data."""

    name = "TrendAgent"

    async def analyse(self, snapshot: MarketSnapshot, quant_summary: str = "") -> AgentResult:
        t0 = time.perf_counter()
        cfg = self.settings.trading.timeframes
        context_tf = cfg["context"]
        primary_tf = cfg["primary"]

        context_data = _ohlcv_text(snapshot, context_tf)
        primary_data = _ohlcv_text(snapshot, primary_tf)

        user_msg = PromptLibrary.render(
            PromptLibrary.TREND_ANALYSIS,
            symbol=snapshot.symbol,
            context_tf=context_tf,
            context_data=context_data,
            primary_tf=primary_tf,
            primary_data=primary_data,
            indicators=quant_summary,
        )

        try:
            data = await self._ask(PromptLibrary.SYSTEM_TREND_AGENT, user_msg, snapshot.symbol)
            # Normalise direction field
            trend = data.get("trend", "sideways").lower()
            data["direction"] = "long" if trend == "bullish" else "short" if trend == "bearish" else "none"
            elapsed = (time.perf_counter() - t0) * 1000
            return self._make_result(snapshot.symbol, success=True, data=data, elapsed_ms=elapsed)
        except Exception as exc:
            logger.error(f"TrendAgent error for {snapshot.symbol}: {exc}")
            return self._make_result(snapshot.symbol, success=False, data={}, error=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Setup Agent
# ══════════════════════════════════════════════════════════════════════════════

class SetupAgent(_LLMAgent):
    """Evaluates trade setup quality given trend context."""

    name = "SetupAgent"

    async def analyse(
        self,
        snapshot: MarketSnapshot,
        trend_result: AgentResult,
        quant_summary: str = "",
    ) -> AgentResult:
        t0 = time.perf_counter()
        cfg = self.settings.trading.timeframes
        primary_tf = cfg["primary"]

        trend_summary = _format_trend_summary(trend_result)
        primary_data = _ohlcv_text(snapshot, primary_tf)

        user_msg = PromptLibrary.render(
            PromptLibrary.SETUP_ANALYSIS,
            symbol=snapshot.symbol,
            trend_summary=trend_summary,
            primary_tf=primary_tf,
            primary_data=primary_data,
            indicators=quant_summary,
        )

        try:
            data = await self._ask(PromptLibrary.SYSTEM_SETUP_AGENT, user_msg, snapshot.symbol)
            if "direction" not in data:
                data["direction"] = "none"
            elapsed = (time.perf_counter() - t0) * 1000
            return self._make_result(snapshot.symbol, success=True, data=data, elapsed_ms=elapsed)
        except Exception as exc:
            logger.error(f"SetupAgent error for {snapshot.symbol}: {exc}")
            return self._make_result(snapshot.symbol, success=False, data={}, error=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Trigger Agent
# ══════════════════════════════════════════════════════════════════════════════

class TriggerAgent(_LLMAgent):
    """Identifies precise entry triggers on the lowest timeframe."""

    name = "TriggerAgent"

    async def analyse(
        self,
        snapshot: MarketSnapshot,
        setup_result: AgentResult,
    ) -> AgentResult:
        t0 = time.perf_counter()
        entry_tf = self.settings.trading.timeframes["entry"]
        setup_summary = _format_setup_summary(setup_result)
        entry_data = _ohlcv_text(snapshot, entry_tf)
        current_price = snapshot.current_price

        user_msg = PromptLibrary.render(
            PromptLibrary.TRIGGER_ANALYSIS,
            symbol=snapshot.symbol,
            entry_tf=entry_tf,
            setup_summary=setup_summary,
            entry_data=entry_data,
            current_price=str(current_price),
        )

        try:
            data = await self._ask(PromptLibrary.SYSTEM_TRIGGER_AGENT, user_msg, snapshot.symbol)
            if not data.get("trigger_fired", False):
                data["direction"] = "none"
            else:
                direction = setup_result.data.get("direction", "none")
                data["direction"] = direction
            elapsed = (time.perf_counter() - t0) * 1000
            return self._make_result(snapshot.symbol, success=True, data=data, elapsed_ms=elapsed)
        except Exception as exc:
            logger.error(f"TriggerAgent error for {snapshot.symbol}: {exc}")
            return self._make_result(snapshot.symbol, success=False, data={}, error=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# 4. Sentiment Agent
# ══════════════════════════════════════════════════════════════════════════════

class SentimentAgent(_LLMAgent):
    """Interprets news and social signals to produce a sentiment score."""

    name = "SentimentAgent"

    async def analyse(self, snapshot: MarketSnapshot) -> AgentResult:
        t0 = time.perf_counter()
        hours = self.settings.data.sentiment_lookback_hours
        headlines = snapshot.news_headlines_text()
        social_signals = str(snapshot.social_signals) if snapshot.social_signals else "N/A"
        alt_data_text = snapshot.alt_data_text()

        user_msg = PromptLibrary.render(
            PromptLibrary.SENTIMENT_ANALYSIS,
            symbol=snapshot.symbol,
            hours=str(hours),
            headlines=headlines,
            alt_data=alt_data_text,
            social_signals=social_signals,
        )

        try:
            data = await self._ask(PromptLibrary.SYSTEM_SENTIMENT_AGENT, user_msg, snapshot.symbol)
            sentiment = data.get("sentiment", "neutral").lower()
            data["direction"] = "long" if sentiment == "bullish" else "short" if sentiment == "bearish" else "none"
            elapsed = (time.perf_counter() - t0) * 1000
            return self._make_result(snapshot.symbol, success=True, data=data, elapsed_ms=elapsed)
        except Exception as exc:
            logger.error(f"SentimentAgent error for {snapshot.symbol}: {exc}")
            return self._make_result(
                snapshot.symbol,
                success=False,
                data={"direction": "none", "confidence": 0.0, "score": 0.0},
                error=str(exc),
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ohlcv_text(snapshot: MarketSnapshot, tf: str, n: int = 10) -> str:
    ohlcv = snapshot.ohlcv.get(tf)
    if ohlcv is None:
        return f"No data for {tf} timeframe."
    return ohlcv.summary(n=n)


def _format_trend_summary(r: AgentResult) -> str:
    if not r.success:
        return "Trend analysis unavailable."
    d = r.data
    return (
        f"Trend: {d.get('trend', 'unknown')} (strength {d.get('strength', 'N/A')}/10)\n"
        f"Reasoning: {d.get('reasoning', '')}\n"
        f"Key levels: {d.get('key_levels', {})}"
    )


def _format_setup_summary(r: AgentResult) -> str:
    if not r.success:
        return "Setup analysis unavailable."
    d = r.data
    return (
        f"Setup: {d.get('setup_type', 'unknown')} | Direction: {d.get('direction', 'none')}\n"
        f"Quality: {d.get('quality', 'N/A')}/10\n"
        f"Invalidation: {d.get('invalidation_price', 'N/A')}\n"
        f"Target: {d.get('target_price', 'N/A')}\n"
        f"Reasoning: {d.get('reasoning', '')}"
    )

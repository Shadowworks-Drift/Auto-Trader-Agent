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
from llm.finbert_client import FinBERTClient
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
    """
    Two-tier sentiment agent.

    Tier 1 (fast, <50ms): FinBERT2 classifier on individual headlines
    Tier 2 (slow, 2-15s): Ollama LLM for ambiguous cases and structured
                          reasoning with alternative data

    If FinBERT2 aggregate confidence > threshold AND |score| > ambiguity threshold,
    returns immediately without calling Ollama (~60% of cases).
    Otherwise escalates to full Ollama prompt for richer context integration.
    """

    name = "SentimentAgent"
    FAST_PATH_CONF_THRESHOLD = 0.70    # FinBERT must be this confident to skip Ollama
    FAST_PATH_SCORE_THRESHOLD = 0.35   # |score| must exceed this to trust fast path

    def __init__(self, settings: Settings, llm: OllamaClient, finbert: Optional[FinBERTClient] = None) -> None:
        super().__init__(settings, llm)
        self.finbert = finbert

    async def analyse(self, snapshot: MarketSnapshot) -> AgentResult:
        t0 = time.perf_counter()
        hours = self.settings.data.sentiment_lookback_hours

        # ── Tier 1: FinBERT2 fast-path ────────────────────────────────────
        finbert_score = None
        if self.finbert and self.finbert.is_available() and snapshot.news:
            headline_texts = [item.title for item in snapshot.news[:15]]
            try:
                scores = await self.finbert.score_headlines(headline_texts)
                finbert_score = self.finbert.aggregate(scores)
                logger.debug(
                    f"FinBERT2 [{snapshot.symbol}]: {finbert_score.direction} "
                    f"score={finbert_score.score:.3f} conf={finbert_score.confidence:.3f} "
                    f"({finbert_score.elapsed_ms:.0f}ms)"
                )
                # Fast-path exit: confident and unambiguous
                if (
                    finbert_score.confidence >= self.FAST_PATH_CONF_THRESHOLD
                    and abs(finbert_score.score) >= self.FAST_PATH_SCORE_THRESHOLD
                    and snapshot.alt_composite_signal is None  # no alt data to integrate
                ):
                    direction = "long" if finbert_score.direction == "bullish" else \
                                "short" if finbert_score.direction == "bearish" else "none"
                    elapsed = (time.perf_counter() - t0) * 1000
                    return self._make_result(snapshot.symbol, success=True, data={
                        "sentiment": finbert_score.direction,
                        "score": finbert_score.score,
                        "confidence": finbert_score.confidence,
                        "direction": direction,
                        "momentum": "stable",
                        "reasoning": f"FinBERT2 fast-path: {finbert_score.label} ({finbert_score.score:.3f})",
                        "model_used": finbert_score.model_used,
                    }, elapsed_ms=elapsed)
            except Exception as exc:
                logger.debug(f"FinBERT2 fast-path failed: {exc}")

        # ── Tier 2: Ollama slow-path (full context) ───────────────────────
        headlines = snapshot.news_headlines_text()
        alt_data_text = snapshot.alt_data_text() if hasattr(snapshot, "alt_data_text") else "N/A"
        social_signals = str(snapshot.social_signals) if snapshot.social_signals else "N/A"

        # Inject FinBERT pre-score into context for Ollama
        if finbert_score:
            headlines = (
                f"[FinBERT2 pre-score: {finbert_score.label} ({finbert_score.score:+.3f})]\n\n"
                + headlines
            )

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
            data["model_used"] = "ollama"
            if finbert_score:
                data["finbert_pre_score"] = finbert_score.score
            elapsed = (time.perf_counter() - t0) * 1000
            return self._make_result(snapshot.symbol, success=True, data=data, elapsed_ms=elapsed)
        except Exception as exc:
            logger.error(f"SentimentAgent error for {snapshot.symbol}: {exc}")
            # If FinBERT2 ran and we have a result, use it as fallback
            if finbert_score:
                direction = "long" if finbert_score.direction == "bullish" else \
                            "short" if finbert_score.direction == "bearish" else "none"
                return self._make_result(snapshot.symbol, success=True, data={
                    "sentiment": finbert_score.direction,
                    "score": finbert_score.score,
                    "confidence": finbert_score.confidence * 0.7,  # penalise for LLM failure
                    "direction": direction,
                    "reasoning": f"FinBERT2 fallback (Ollama failed): {exc}",
                    "model_used": "finbert2_fallback",
                })
            return self._make_result(
                snapshot.symbol, success=False,
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

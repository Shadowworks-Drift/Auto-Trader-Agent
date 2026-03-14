"""
Decision Core — Adversarial Decision Framework.

Fuses signals from Quant, Trend, Setup, Trigger, and Sentiment agents using
configurable weights.  When adversarial mode is on, a Bear or Bull advocate
LLM agent challenges the proposal before the final confidence score is computed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from config.settings import Settings
from data.market_data import MarketSnapshot
from llm.ollama_client import OllamaClient
from llm.prompts import PromptLibrary
from .base_agent import AgentResult
from .regime_detector import RegimeResult, REGIME_WEIGHT_ADJUSTMENTS

try:
    from rl.position_sizer import RLPositionSizer, SizerDecision
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


@dataclass
class TradeProposal:
    symbol: str
    direction: str           # "long" | "short" | "none"
    confidence: float        # 0.0 – 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    agent_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    adversarial_summary: str = ""
    sizing_method: str = "kelly_lite"  # "rl_ppo" | "half_kelly" | "kelly_lite"
    regime: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def risk_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return abs(self.entry_price - self.stop_loss) / self.entry_price

    @property
    def reward_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return abs(self.take_profit - self.entry_price) / self.entry_price

    @property
    def risk_reward(self) -> float:
        return self.reward_pct / max(self.risk_pct, 1e-9)

    def is_actionable(self, min_confidence: float) -> bool:
        return self.direction != "none" and self.confidence >= min_confidence


class DecisionCore:
    """
    Fuses all agent outputs into a single TradeProposal.

    Weights (configurable in settings.decision):
      quant   40%  |  trend  20%  |  setup  20%  |  trigger  10%  |  sentiment  10%
    """

    def __init__(
        self,
        settings: Settings,
        llm: OllamaClient,
        rl_sizer: Optional["RLPositionSizer"] = None,
    ) -> None:
        self.settings = settings
        self.cfg = settings.decision
        self.risk_cfg = settings.risk
        self.llm = llm
        self.rl_sizer = rl_sizer  # Optional RL position sizer

    async def decide(
        self,
        snapshot: MarketSnapshot,
        quant_result: AgentResult,
        trend_result: AgentResult,
        setup_result: AgentResult,
        trigger_result: AgentResult,
        sentiment_result: AgentResult,
        regime_result: Optional[RegimeResult] = None,
        vision_result: Optional[AgentResult] = None,
    ) -> TradeProposal:
        t0 = time.perf_counter()

        # ── Step 1: directional vote ───────────────────────────────────────
        all_results = [quant_result, trend_result, setup_result, trigger_result, sentiment_result]
        if vision_result is not None:
            all_results.append(vision_result)
        direction = self._majority_direction(all_results)

        if direction == "none":
            return TradeProposal(
                symbol=snapshot.symbol,
                direction="none",
                confidence=0.0,
                entry_price=snapshot.current_price,
                stop_loss=0.0,
                take_profit=0.0,
                position_size_pct=0.0,
                reasoning="No directional consensus across agents.",
            )

        # ── Regime gate ───────────────────────────────────────────────────
        if regime_result and not regime_result.is_tradeable:
            return TradeProposal(
                symbol=snapshot.symbol,
                direction="none",
                confidence=0.0,
                entry_price=snapshot.current_price,
                stop_loss=0.0,
                take_profit=0.0,
                position_size_pct=0.0,
                regime=regime_result.regime,
                reasoning=f"Regime gate blocked trade: {regime_result.reasoning}",
            )

        # ── Step 2: fuse confidence scores (regime-aware) ─────────────────
        score = self._fuse_scores(
            direction, quant_result, trend_result, setup_result, trigger_result, sentiment_result,
            regime=regime_result,
            vision=vision_result,
        )

        # ── Step 3: compute entry / SL / TP ───────────────────────────────
        price = snapshot.current_price
        entry, sl, tp = self._compute_levels(direction, price, trigger_result, setup_result)

        # ── Step 4: adversarial debate ────────────────────────────────────
        adversarial_summary = ""
        if self.cfg.adversarial_veto and score >= self.cfg.min_confidence * 0.8:
            adv_discount, adversarial_summary = await self._run_adversarial(
                snapshot, direction, entry, sl, tp, quant_result, trend_result
            )
            score = max(0.0, score - adv_discount)

        # ── Step 5: position sizing (regime-adjusted) ─────────────────────
        position_size_pct, sizing_method = self._size_position(
            score, entry, sl,
            quant_result, trend_result, setup_result, trigger_result, sentiment_result,
            snapshot,
        )
        if regime_result:
            position_size_pct *= regime_result.position_size_mult

        all_for_reasoning = [quant_result, trend_result, setup_result, trigger_result, sentiment_result]
        if vision_result is not None:
            all_for_reasoning.append(vision_result)
        reasoning = self._build_reasoning(direction, score, *all_for_reasoning)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"DecisionCore [{snapshot.symbol}] → {direction.upper()} "
            f"confidence={score:.2f} entry={entry:.4f} sl={sl:.4f} tp={tp:.4f} "
            f"({elapsed:.0f}ms)"
        )

        agent_scores = {
            "quant": _dir_score(direction, quant_result),
            "trend": _dir_score(direction, trend_result),
            "setup": _dir_score(direction, setup_result),
            "trigger": _dir_score(direction, trigger_result),
            "sentiment": _dir_score(direction, sentiment_result),
        }
        if vision_result is not None:
            agent_scores["vision"] = _dir_score(direction, vision_result)

        return TradeProposal(
            symbol=snapshot.symbol,
            direction=direction,
            confidence=round(score, 4),
            entry_price=round(entry, 8),
            stop_loss=round(sl, 8),
            take_profit=round(tp, 8),
            position_size_pct=round(position_size_pct, 4),
            regime=regime_result.regime if regime_result else "unknown",
            agent_scores=agent_scores,
            reasoning=reasoning,
            adversarial_summary=adversarial_summary,
            sizing_method=sizing_method,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _majority_direction(self, results: List[AgentResult]) -> str:
        votes: Dict[str, float] = {"long": 0.0, "short": 0.0, "none": 0.0}
        for r in results:
            if r.success:
                d = r.direction
                key = d if d in votes else "none"
                votes[key] += r.confidence
        best = max(votes, key=lambda k: votes[k])
        total_directional = votes["long"] + votes["short"]
        if total_directional < 0.1:
            return "none"
        agreement = votes[best] / max(total_directional, 1e-9)
        if agreement < self.cfg.consensus_threshold:
            return "none"
        return best

    def _fuse_scores(
        self,
        direction: str,
        quant: AgentResult,
        trend: AgentResult,
        setup: AgentResult,
        trigger: AgentResult,
        sentiment: AgentResult,
        regime: Optional[RegimeResult] = None,
        vision: Optional[AgentResult] = None,
    ) -> float:
        w = self.cfg
        # Base weights
        qw = w.quant_weight
        trw = w.trend_weight
        sw = w.setup_weight
        trgw = w.trigger_weight
        sentw = w.sentiment_weight
        # Vision weight: only applied when the agent ran successfully
        visw = w.vision_weight if (vision is not None and vision.success) else 0.0

        # Apply regime-conditional weight adjustments
        if regime and regime.weight_adjustments:
            adj = regime.weight_adjustments
            # Quant sub-signals adjust via their indicator votes
            qw   *= adj.get("ema_trend", 1.0) * 0.5 + adj.get("macd", 1.0) * 0.5
            trw  *= adj.get("trend", 1.0)
            sw   *= adj.get("setup", 1.0)
            trgw *= adj.get("trigger", 1.0)
            sentw *= adj.get("sentiment", 1.0)

        weighted = (
            _dir_score(direction, quant)       * qw
            + _dir_score(direction, trend)     * trw
            + _dir_score(direction, setup)     * sw
            + _dir_score(direction, trigger)   * trgw
            + _dir_score(direction, sentiment) * sentw
        )
        if visw > 0.0 and vision is not None:
            weighted += _dir_score(direction, vision) * visw
        total_w = qw + trw + sw + trgw + sentw + visw
        return round(weighted / max(total_w, 1e-9), 4)

    def _compute_levels(
        self,
        direction: str,
        price: float,
        trigger: AgentResult,
        setup: AgentResult,
    ) -> tuple:
        # Prefer LLM-suggested levels when available and plausible
        entry = float(trigger.data.get("entry_price") or price)
        sl = float(trigger.data.get("stop_loss") or setup.data.get("invalidation_price") or 0)
        tp = float(trigger.data.get("take_profit") or setup.data.get("target_price") or 0)

        # Fallback to percentage-based levels
        sl_pct = self.risk_cfg.stop_loss_pct
        tp_pct = self.risk_cfg.take_profit_pct
        if entry <= 0:
            entry = price
        if direction == "long":
            if sl <= 0 or sl >= entry:
                sl = entry * (1 - sl_pct)
            if tp <= 0 or tp <= entry:
                tp = entry * (1 + tp_pct)
        else:
            if sl <= 0 or sl <= entry:
                sl = entry * (1 + sl_pct)
            if tp <= 0 or tp >= entry:
                tp = entry * (1 - tp_pct)

        return entry, sl, tp

    def _size_position(
        self,
        confidence: float,
        entry: float,
        sl: float,
        quant_result: AgentResult,
        trend_result: AgentResult,
        setup_result: AgentResult,
        trigger_result: AgentResult,
        sentiment_result: AgentResult,
        snapshot: "MarketSnapshot",
    ) -> tuple:
        """Position sizing via RL agent (if loaded) or Kelly-lite fallback."""
        base_size = self.settings.trading.position_size_pct

        if RL_AVAILABLE and self.rl_sizer is not None:
            try:
                decision = self.rl_sizer.decide_from_dict(
                    quant_data=quant_result.data,
                    trend_data=trend_result.data,
                    setup_data=setup_result.data,
                    trigger_data=trigger_result.data,
                    sentiment_data=sentiment_result.data,
                    portfolio_info={
                        "funding_rate": snapshot.alt_data.funding.rate
                        if hasattr(snapshot, "alt_data") and snapshot.alt_data and snapshot.alt_data.funding
                        else 0.0,
                        "fear_greed": snapshot.alt_data.fear_greed.value
                        if hasattr(snapshot, "alt_data") and snapshot.alt_data and snapshot.alt_data.fear_greed
                        else 50,
                        "drawdown": 0.0,
                        "daily_pnl_pct": 0.0,
                        "open_count": 0,
                        "max_positions": self.settings.risk.max_open_positions
                        if hasattr(self.settings.risk, "max_open_positions") else 3,
                    },
                )
                logger.debug(
                    f"RL sizer [{snapshot.symbol}]: size={decision.position_size_pct:.4f} "
                    f"method={decision.method} raw={decision.raw_action:.3f}"
                )
                return decision.position_size_pct, decision.method
            except Exception as exc:
                logger.warning(f"RL sizer failed, using Kelly-lite: {exc}")

        # Kelly-lite fallback
        size = min(base_size * confidence, base_size)
        return size, "kelly_lite"

    async def _run_adversarial(
        self,
        snapshot: MarketSnapshot,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        quant: AgentResult,
        trend: AgentResult,
    ) -> tuple:
        """Run a bear or bull advocate and return (confidence_discount, summary)."""
        market_data = (
            quant.data.get("summary_text", "")
            + "\n"
            + trend.data.get("reasoning", "")
        )
        try:
            if direction == "long":
                bull_summary = trend.data.get("reasoning", "Bullish trend identified.")
                user_msg = PromptLibrary.render(
                    PromptLibrary.ADVERSARIAL_BEAR,
                    symbol=snapshot.symbol,
                    entry_price=str(entry),
                    stop_loss=str(sl),
                    take_profit=str(tp),
                    bull_summary=bull_summary,
                    market_data=market_data,
                )
                system = PromptLibrary.SYSTEM_BEAR_ADVOCATE
            else:
                bear_summary = trend.data.get("reasoning", "Bearish trend identified.")
                user_msg = PromptLibrary.render(
                    PromptLibrary.ADVERSARIAL_BULL,
                    symbol=snapshot.symbol,
                    entry_price=str(entry),
                    stop_loss=str(sl),
                    take_profit=str(tp),
                    bear_summary=bear_summary,
                    market_data=market_data,
                )
                system = PromptLibrary.SYSTEM_BULL_ADVOCATE

            resp = await self.llm.chat(
                messages=[{"role": "user", "content": user_msg}],
                system=system,
                expect_json=True,
            )
            parsed = resp.parsed or {}
            conviction_key = "overall_bearish_conviction" if direction == "long" else "overall_bullish_conviction"
            adv_conviction = float(parsed.get(conviction_key, 0.3))
            # Discount confidence proportional to adversarial conviction
            discount = adv_conviction * 0.25
            args = parsed.get("bear_arguments" if direction == "long" else "bull_arguments", [])
            summary = f"Adversarial ({direction} challenged):\n" + "\n".join(f"  - {a}" for a in args[:5])
            return discount, summary
        except Exception as exc:
            logger.warning(f"Adversarial debate failed: {exc}")
            return 0.0, ""

    def _build_reasoning(self, direction: str, score: float, *results: AgentResult) -> str:
        lines = [f"Decision: {direction.upper()} | Confidence: {score:.2f}"]
        for r in results:
            if r.success:
                reasoning = r.data.get("reasoning", "")[:150]
                lines.append(f"  [{r.agent_name}] {reasoning}")
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dir_score(direction: str, result: AgentResult) -> float:
    """Score [0,1] representing how much this agent supports the chosen direction."""
    if not result.success:
        return 0.0
    if result.direction == direction:
        return result.confidence
    if result.direction == "none":
        return result.confidence * 0.5
    return 0.0  # opposing direction = 0

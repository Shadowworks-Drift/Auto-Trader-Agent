"""
Risk Audit — final gate before any order reaches the exchange.

Checks:
  1. Stop-loss direction is correct vs entry
  2. Risk:Reward ratio meets minimum
  3. Daily loss circuit-breaker
  4. Portfolio drawdown circuit-breaker
  5. Maximum open positions
  6. Correlated position concentration
  7. LLM-based narrative risk review (optional)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from config.settings import Settings
from llm.ollama_client import OllamaClient
from llm.prompts import PromptLibrary
from .decision_core import TradeProposal


@dataclass
class RiskDecision:
    approved: bool
    veto_reason: Optional[str]
    adjusted_stop_loss: Optional[float]
    adjusted_take_profit: Optional[float]
    adjusted_size_pct: Optional[float]
    risk_score: float   # 0=safe, 1=very risky
    reasoning: str
    timestamp: datetime = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class PortfolioState:
    """Tracks live portfolio state consumed by RiskAudit."""

    def __init__(self, initial_value: float = 10_000.0) -> None:
        self.initial_value = initial_value
        self.current_value = initial_value
        self.peak_value = initial_value
        self.daily_start_value = initial_value
        self.open_positions: List[Dict[str, Any]] = []

    @property
    def drawdown_pct(self) -> float:
        return (self.peak_value - self.current_value) / max(self.peak_value, 1e-9)

    @property
    def daily_pnl_pct(self) -> float:
        return (self.current_value - self.daily_start_value) / max(self.daily_start_value, 1e-9)

    @property
    def open_position_count(self) -> int:
        return len(self.open_positions)

    def update_value(self, new_value: float) -> None:
        self.current_value = new_value
        if new_value > self.peak_value:
            self.peak_value = new_value

    def add_position(self, position: Dict[str, Any]) -> None:
        self.open_positions.append(position)

    def remove_position(self, symbol: str) -> None:
        self.open_positions = [p for p in self.open_positions if p.get("symbol") != symbol]

    def has_open_position(self, symbol: str) -> bool:
        return any(p.get("symbol") == symbol for p in self.open_positions)

    def correlated_count(self, symbol: str) -> int:
        """Count how many open positions are in the same base currency group."""
        base = symbol.split("/")[0].upper()
        btc_alts = {"BTC", "ETH", "BNB", "SOL", "AVAX", "MATIC", "DOT", "ADA"}
        if base in btc_alts:
            return sum(1 for p in self.open_positions if p.get("symbol", "").split("/")[0].upper() in btc_alts)
        return 0


class RiskAudit:
    """
    Validates a TradeProposal against hard risk limits.

    Hard limits (rule-based) are evaluated first.
    If proposal survives, an optional LLM audit adds a narrative layer.
    """

    def __init__(self, settings: Settings, portfolio: PortfolioState, llm: Optional[OllamaClient] = None) -> None:
        self.cfg = settings.risk
        self.decision_cfg = settings.decision
        self.portfolio = portfolio
        self.llm = llm

    async def audit(self, proposal: TradeProposal, agent_signals: str = "") -> RiskDecision:
        t0 = time.perf_counter()

        # ── Hard rules ─────────────────────────────────────────────────────
        veto, reason = self._hard_checks(proposal)
        if veto:
            return RiskDecision(
                approved=False,
                veto_reason=reason,
                adjusted_stop_loss=None,
                adjusted_take_profit=None,
                adjusted_size_pct=None,
                risk_score=1.0,
                reasoning=f"Hard veto: {reason}",
            )

        # ── Soft adjustments ───────────────────────────────────────────────
        adj_sl = self._correct_stop_loss(proposal)
        adj_tp = proposal.take_profit
        adj_size = self._adjust_size(proposal)

        # ── LLM narrative audit ────────────────────────────────────────────
        risk_score = self._compute_risk_score(proposal)
        reasoning = "Passed all hard risk checks."

        if self.llm and proposal.confidence >= self.decision_cfg.min_confidence:
            try:
                llm_decision = await self._llm_audit(proposal, adj_sl, adj_tp, adj_size, agent_signals)
                if not llm_decision.get("approved", True):
                    return RiskDecision(
                        approved=False,
                        veto_reason=llm_decision.get("veto_reason"),
                        adjusted_stop_loss=None,
                        adjusted_take_profit=None,
                        adjusted_size_pct=None,
                        risk_score=float(llm_decision.get("risk_score", 0.8)),
                        reasoning=llm_decision.get("reasoning", "LLM veto"),
                    )
                adj_sl = llm_decision.get("adjusted_stop_loss") or adj_sl
                adj_tp = llm_decision.get("adjusted_take_profit") or adj_tp
                adj_size = llm_decision.get("adjusted_size_pct") or adj_size
                risk_score = float(llm_decision.get("risk_score", risk_score))
                reasoning = llm_decision.get("reasoning", reasoning)
            except Exception as exc:
                logger.warning(f"LLM risk audit failed (using rule-based): {exc}")

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"RiskAudit [{proposal.symbol}] APPROVED "
            f"risk_score={risk_score:.2f} ({elapsed:.0f}ms)"
        )
        return RiskDecision(
            approved=True,
            veto_reason=None,
            adjusted_stop_loss=adj_sl,
            adjusted_take_profit=adj_tp,
            adjusted_size_pct=adj_size,
            risk_score=risk_score,
            reasoning=reasoning,
        )

    # ── Hard checks ────────────────────────────────────────────────────────────

    def _hard_checks(self, p: TradeProposal) -> tuple:
        """Return (should_veto: bool, reason: str)."""

        # Portfolio drawdown circuit-breaker
        if self.portfolio.drawdown_pct >= self.cfg.max_portfolio_drawdown_pct:
            return True, (
                f"Portfolio drawdown {self.portfolio.drawdown_pct:.1%} "
                f">= limit {self.cfg.max_portfolio_drawdown_pct:.1%}"
            )

        # Daily loss circuit-breaker
        if self.portfolio.daily_pnl_pct <= -self.cfg.max_daily_loss_pct:
            return True, (
                f"Daily loss {self.portfolio.daily_pnl_pct:.1%} "
                f"exceeds limit {self.cfg.max_daily_loss_pct:.1%}"
            )

        # Maximum open positions
        from config.settings import get_settings
        max_pos = get_settings().trading.max_open_positions
        if self.portfolio.open_position_count >= max_pos:
            return True, f"Max open positions ({max_pos}) reached"

        # Already in this symbol
        if self.portfolio.has_open_position(p.symbol):
            return True, f"Already holding an open position in {p.symbol}"

        # R:R check
        if p.risk_reward < self.cfg.min_risk_reward_ratio:
            return True, (
                f"R:R {p.risk_reward:.2f} < minimum {self.cfg.min_risk_reward_ratio}"
            )

        # SL direction sanity
        if p.direction == "long" and p.stop_loss >= p.entry_price:
            return True, "Stop-loss >= entry price for a LONG trade"
        if p.direction == "short" and p.stop_loss <= p.entry_price:
            return True, "Stop-loss <= entry price for a SHORT trade"

        # Correlated position concentration
        corr = self.portfolio.correlated_count(p.symbol)
        if corr >= self.cfg.max_correlated_positions:
            return True, f"Too many correlated positions ({corr})"

        return False, ""

    def _correct_stop_loss(self, p: TradeProposal) -> float:
        """Ensure SL is at least stop_loss_pct from entry in correct direction."""
        if p.direction == "long":
            min_sl = p.entry_price * (1 - self.cfg.stop_loss_pct)
            return min(p.stop_loss, min_sl)
        else:
            max_sl = p.entry_price * (1 + self.cfg.stop_loss_pct)
            return max(p.stop_loss, max_sl)

    def _adjust_size(self, p: TradeProposal) -> float:
        """Kelly-scaling: reduce size if risk_pct > stop_loss_pct."""
        if p.risk_pct > self.cfg.stop_loss_pct * 2:
            scale = self.cfg.stop_loss_pct / max(p.risk_pct, 1e-9)
            return round(p.position_size_pct * scale, 4)
        return p.position_size_pct

    def _compute_risk_score(self, p: TradeProposal) -> float:
        """Heuristic 0-1 risk score."""
        rr_score = max(0.0, 1.0 - (p.risk_reward - 1.5) / 3.0)
        draw_score = self.portfolio.drawdown_pct / max(self.cfg.max_portfolio_drawdown_pct, 1e-9)
        conf_score = 1.0 - p.confidence
        return round(min(0.5 * rr_score + 0.3 * draw_score + 0.2 * conf_score, 1.0), 4)

    async def _llm_audit(
        self,
        p: TradeProposal,
        adj_sl: float,
        adj_tp: float,
        adj_size: float,
        agent_signals: str,
    ) -> Dict[str, Any]:
        portfolio = self.portfolio
        user_msg = PromptLibrary.render(
            PromptLibrary.RISK_AUDIT,
            symbol=p.symbol,
            direction=p.direction,
            entry_price=str(p.entry_price),
            stop_loss=str(adj_sl),
            risk_pct=f"{p.risk_pct * 100:.2f}",
            take_profit=str(adj_tp),
            reward_pct=f"{p.reward_pct * 100:.2f}",
            risk_reward=f"{p.risk_reward:.2f}",
            position_size_pct=f"{adj_size * 100:.2f}",
            portfolio_value=str(portfolio.current_value),
            open_positions=str(portfolio.open_position_count),
            daily_pnl_pct=f"{portfolio.daily_pnl_pct * 100:.2f}",
            max_drawdown_pct=f"{self.cfg.max_portfolio_drawdown_pct * 100:.1f}",
            agent_signals=agent_signals,
            adversarial_args=p.adversarial_summary,
            min_rr=str(self.cfg.min_risk_reward_ratio),
        )
        resp = await self.llm.chat(
            messages=[{"role": "user", "content": user_msg}],
            system=PromptLibrary.SYSTEM_RISK_AGENT,
            expect_json=True,
        )
        return resp.parsed or {"approved": True, "risk_score": 0.5, "reasoning": resp.content[:300]}

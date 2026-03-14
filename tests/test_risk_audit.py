"""
Tests for RiskAudit hard-rule checks.
"""

from __future__ import annotations

import pytest

from agents.decision_core import TradeProposal
from agents.risk_audit import RiskAudit, PortfolioState
from config.settings import Settings


def _make_proposal(**overrides) -> TradeProposal:
    base = dict(
        symbol="BTC/USDT",
        direction="long",
        confidence=0.75,
        entry_price=40000.0,
        stop_loss=38800.0,    # 3% below
        take_profit=42400.0,  # 6% above
        position_size_pct=0.05,
    )
    base.update(overrides)
    return TradeProposal(**base)


@pytest.fixture
def portfolio() -> PortfolioState:
    return PortfolioState(initial_value=10_000.0)


@pytest.mark.asyncio
async def test_valid_proposal_approved(settings: Settings, portfolio: PortfolioState) -> None:
    audit = RiskAudit(settings, portfolio, llm=None)
    proposal = _make_proposal()
    decision = await audit.audit(proposal)
    assert decision.approved


@pytest.mark.asyncio
async def test_sl_above_entry_vetoed(settings: Settings, portfolio: PortfolioState) -> None:
    audit = RiskAudit(settings, portfolio, llm=None)
    proposal = _make_proposal(stop_loss=41000.0)  # SL above entry for long
    decision = await audit.audit(proposal)
    assert not decision.approved
    assert decision.veto_reason is not None


@pytest.mark.asyncio
async def test_poor_rr_vetoed(settings: Settings, portfolio: PortfolioState) -> None:
    audit = RiskAudit(settings, portfolio, llm=None)
    # Very tight TP, wide SL → R:R < 1.5
    proposal = _make_proposal(
        entry_price=40000.0,
        stop_loss=38000.0,   # 5% risk
        take_profit=40400.0, # 1% reward → R:R = 0.2
    )
    decision = await audit.audit(proposal)
    assert not decision.approved


@pytest.mark.asyncio
async def test_drawdown_circuit_breaker(settings: Settings) -> None:
    portfolio = PortfolioState(initial_value=10_000.0)
    portfolio.update_value(8_000.0)  # 20% drawdown
    audit = RiskAudit(settings, portfolio, llm=None)
    proposal = _make_proposal()
    decision = await audit.audit(proposal)
    assert not decision.approved
    assert "drawdown" in decision.veto_reason.lower()


@pytest.mark.asyncio
async def test_max_positions_circuit_breaker(settings: Settings) -> None:
    portfolio = PortfolioState(initial_value=10_000.0)
    for i in range(settings.trading.max_open_positions):
        portfolio.add_position({"symbol": f"COIN{i}/USDT"})
    audit = RiskAudit(settings, portfolio, llm=None)
    proposal = _make_proposal(symbol="NEW/USDT")
    decision = await audit.audit(proposal)
    assert not decision.approved


@pytest.mark.asyncio
async def test_duplicate_position_vetoed(settings: Settings) -> None:
    portfolio = PortfolioState(initial_value=10_000.0)
    portfolio.add_position({"symbol": "BTC/USDT"})
    audit = RiskAudit(settings, portfolio, llm=None)
    proposal = _make_proposal(symbol="BTC/USDT")
    decision = await audit.audit(proposal)
    assert not decision.approved

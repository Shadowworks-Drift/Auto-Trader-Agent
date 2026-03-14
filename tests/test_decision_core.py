"""
Tests for DecisionCore signal fusion.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from agents.base_agent import AgentResult
from agents.decision_core import DecisionCore, TradeProposal, _dir_score
from config.settings import Settings


def _agent_result(name: str, symbol: str, direction: str, confidence: float) -> AgentResult:
    return AgentResult(
        agent_name=name,
        symbol=symbol,
        timestamp=datetime.utcnow(),
        success=True,
        data={"direction": direction, "confidence": confidence, "reasoning": "test"},
    )


def _failed_result(name: str, symbol: str) -> AgentResult:
    return AgentResult(
        agent_name=name,
        symbol=symbol,
        timestamp=datetime.utcnow(),
        success=False,
        data={},
        error="test error",
    )


def test_dir_score_matching() -> None:
    result = _agent_result("quant", "BTC/USDT", "long", 0.8)
    assert _dir_score("long", result) == pytest.approx(0.8)


def test_dir_score_opposing() -> None:
    result = _agent_result("quant", "BTC/USDT", "short", 0.8)
    assert _dir_score("long", result) == 0.0


def test_dir_score_neutral() -> None:
    result = _agent_result("quant", "BTC/USDT", "none", 0.8)
    assert _dir_score("long", result) == pytest.approx(0.4)


def test_dir_score_failed() -> None:
    result = _failed_result("quant", "BTC/USDT")
    assert _dir_score("long", result) == 0.0


def test_trade_proposal_rr() -> None:
    proposal = TradeProposal(
        symbol="BTC/USDT",
        direction="long",
        confidence=0.75,
        entry_price=40000.0,
        stop_loss=38800.0,   # 3% risk
        take_profit=42400.0, # 6% reward
        position_size_pct=0.05,
    )
    assert proposal.risk_pct == pytest.approx(0.03, rel=1e-2)
    assert proposal.reward_pct == pytest.approx(0.06, rel=1e-2)
    assert proposal.risk_reward == pytest.approx(2.0, rel=1e-2)


def test_trade_proposal_is_actionable() -> None:
    p = TradeProposal("BTC/USDT", "long", 0.75, 40000, 38800, 42400, 0.05)
    assert p.is_actionable(0.65)
    assert not p.is_actionable(0.80)


def test_trade_proposal_none_not_actionable() -> None:
    p = TradeProposal("BTC/USDT", "none", 0.80, 40000, 0, 0, 0)
    assert not p.is_actionable(0.65)

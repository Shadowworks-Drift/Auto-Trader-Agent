"""
Tests for PaperTradingEngine.
"""

from __future__ import annotations

import pytest

from agents.decision_core import TradeProposal
from execution.paper_trading import PaperTradingEngine


@pytest.fixture
async def engine(tmp_path) -> PaperTradingEngine:
    e = PaperTradingEngine(db_path=str(tmp_path / "test_trades.db"), initial_balance=10_000.0)
    await e.init()
    yield e
    await e.close()


@pytest.mark.asyncio
async def test_open_position(engine: PaperTradingEngine) -> None:
    proposal = TradeProposal(
        symbol="BTC/USDT", direction="long", confidence=0.75,
        entry_price=40000.0, stop_loss=38800.0, take_profit=42400.0,
        position_size_pct=0.05,
    )
    pos_id = await engine.open_position(proposal, current_price=40000.0)
    assert pos_id is not None
    assert len(engine.positions) == 1
    assert engine.balance == pytest.approx(10_000.0 * 0.95, rel=1e-4)


@pytest.mark.asyncio
async def test_stop_loss_triggered(engine: PaperTradingEngine) -> None:
    proposal = TradeProposal(
        symbol="BTC/USDT", direction="long", confidence=0.75,
        entry_price=40000.0, stop_loss=38000.0, take_profit=44000.0,
        position_size_pct=0.10,
    )
    await engine.open_position(proposal, current_price=40000.0)
    trades = await engine.update_prices({"BTC/USDT": 37500.0})
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop_loss"
    assert trades[0].pnl < 0
    assert len(engine.positions) == 0


@pytest.mark.asyncio
async def test_take_profit_triggered(engine: PaperTradingEngine) -> None:
    proposal = TradeProposal(
        symbol="BTC/USDT", direction="long", confidence=0.75,
        entry_price=40000.0, stop_loss=38000.0, take_profit=44000.0,
        position_size_pct=0.10,
    )
    await engine.open_position(proposal, current_price=40000.0)
    trades = await engine.update_prices({"BTC/USDT": 45000.0})
    assert len(trades) == 1
    assert trades[0].exit_reason == "take_profit"
    assert trades[0].pnl > 0


@pytest.mark.asyncio
async def test_short_position_stop_loss(engine: PaperTradingEngine) -> None:
    proposal = TradeProposal(
        symbol="ETH/USDT", direction="short", confidence=0.75,
        entry_price=2000.0, stop_loss=2100.0, take_profit=1800.0,
        position_size_pct=0.05,
    )
    await engine.open_position(proposal, current_price=2000.0)
    trades = await engine.update_prices({"ETH/USDT": 2150.0})
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop_loss"
    assert trades[0].pnl < 0


@pytest.mark.asyncio
async def test_win_rate_computed(engine: PaperTradingEngine) -> None:
    for i, (entry, sl, tp) in enumerate(
        [(40000, 38000, 44000), (2000, 2100, 1800), (100, 90, 120)]
    ):
        direction = "long" if i != 1 else "short"
        proposal = TradeProposal(
            symbol=f"COIN{i}/USDT", direction=direction, confidence=0.75,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            position_size_pct=0.05,
        )
        await engine.open_position(proposal, current_price=entry)

    # Close all with profitable exits
    for i, exit_price in [(0, 45000), (1, 1700), (2, 125)]:
        prices = {f"COIN{i}/USDT": exit_price}
        await engine.update_prices(prices)

    assert engine.win_rate() > 0

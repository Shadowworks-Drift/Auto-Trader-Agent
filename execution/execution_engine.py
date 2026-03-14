"""
Execution Engine — unified interface for paper and live trading.

In paper mode: delegates to PaperTradingEngine.
In live mode: places real orders via ccxt with safeguards.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt
from loguru import logger

from agents.decision_core import TradeProposal
from agents.risk_audit import RiskDecision, PortfolioState
from config.settings import Settings
from .paper_trading import PaperTradingEngine, PaperTrade


@dataclass
class OrderResult:
    symbol: str
    direction: str
    order_id: str
    price: float
    amount: float
    status: str         # "filled" | "open" | "failed" | "paper"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class ExecutionEngine:
    """
    Executes approved trade proposals.

    Usage::

        engine = ExecutionEngine(settings)
        await engine.init()
        result = await engine.execute(proposal, risk_decision, current_price)
        await engine.shutdown()
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mode = settings.trading.mode      # "paper" | "live"
        self.paper: Optional[PaperTradingEngine] = None
        self._exchange: Optional[ccxt.Exchange] = None
        self.portfolio = PortfolioState(initial_value=10_000.0)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        if self.mode == "paper":
            self.paper = PaperTradingEngine(
                db_path=self.settings.monitoring.db_path,
                initial_balance=10_000.0,
            )
            await self.paper.init()
        else:
            await self._connect_exchange()
        logger.info(f"ExecutionEngine initialised in {self.mode.upper()} mode")

    async def shutdown(self) -> None:
        if self.paper:
            await self.paper.close()
        if self._exchange:
            await self._exchange.close()

    # ── Main execute ──────────────────────────────────────────────────────────

    async def execute(
        self,
        proposal: TradeProposal,
        risk_decision: RiskDecision,
        current_price: float,
    ) -> OrderResult:
        if not risk_decision.approved:
            return OrderResult(
                symbol=proposal.symbol,
                direction=proposal.direction,
                order_id="vetoed",
                price=current_price,
                amount=0.0,
                status="failed",
                error=risk_decision.veto_reason,
            )

        # Apply risk-adjusted levels
        effective_sl = risk_decision.adjusted_stop_loss or proposal.stop_loss
        effective_tp = risk_decision.adjusted_take_profit or proposal.take_profit
        effective_size = risk_decision.adjusted_size_pct or proposal.position_size_pct

        # Build effective proposal with adjusted values
        effective = TradeProposal(
            symbol=proposal.symbol,
            direction=proposal.direction,
            confidence=proposal.confidence,
            entry_price=current_price,
            stop_loss=effective_sl,
            take_profit=effective_tp,
            position_size_pct=effective_size,
            reasoning=proposal.reasoning,
        )

        if self.mode == "paper":
            return await self._execute_paper(effective, current_price)
        else:
            return await self._execute_live(effective, current_price)

    async def update_positions(self, prices: Dict[str, float]) -> List[PaperTrade]:
        """Update open positions with new prices (paper mode only)."""
        if self.mode == "paper" and self.paper:
            closed = await self.paper.update_prices(prices)
            for trade in closed:
                self.portfolio.remove_position(trade.symbol)
            return closed
        return []

    def get_portfolio_state(self) -> PortfolioState:
        return self.portfolio

    def get_open_positions(self) -> Dict[str, Any]:
        if self.paper:
            return {pos_id: pos.__dict__ for pos_id, pos in self.paper.positions.items()}
        return {}

    def get_performance_summary(self, prices: Dict[str, float]) -> Dict[str, Any]:
        if not self.paper:
            return {}
        return {
            "total_return_pct": self.paper.total_return_pct(prices),
            "sharpe_ratio": self.paper.sharpe_ratio(),
            "max_drawdown": self.paper.max_drawdown(),
            "win_rate": self.paper.win_rate(),
            "total_trades": len(self.paper.trades),
            "open_positions": len(self.paper.positions),
            "portfolio_value": self.paper.portfolio_value(prices),
            "balance": self.paper.balance,
        }

    # ── Paper execution ────────────────────────────────────────────────────────

    async def _execute_paper(self, proposal: TradeProposal, price: float) -> OrderResult:
        pos_id = await self.paper.open_position(proposal, current_price=price)
        if pos_id is None:
            return OrderResult(
                symbol=proposal.symbol,
                direction=proposal.direction,
                order_id="paper_failed",
                price=price,
                amount=0.0,
                status="failed",
                error="Insufficient balance",
            )
        self.portfolio.add_position({"symbol": proposal.symbol, "id": pos_id, "direction": proposal.direction})
        return OrderResult(
            symbol=proposal.symbol,
            direction=proposal.direction,
            order_id=f"paper_{pos_id}",
            price=price,
            amount=proposal.position_size_pct,
            status="paper",
        )

    # ── Live execution ─────────────────────────────────────────────────────────

    async def _execute_live(self, proposal: TradeProposal, price: float) -> OrderResult:
        """Place a real market order + OCO stop-loss/take-profit."""
        if not self._exchange:
            return OrderResult(
                symbol=proposal.symbol, direction=proposal.direction,
                order_id="no_exchange", price=price, amount=0.0,
                status="failed", error="Exchange not connected",
            )
        try:
            balance_info = await self._exchange.fetch_balance()
            quote_balance = float(balance_info["total"].get("USDT", 0))
            trade_value = quote_balance * proposal.position_size_pct
            amount = trade_value / price

            side = "buy" if proposal.direction == "long" else "sell"
            order = await self._exchange.create_order(
                symbol=proposal.symbol,
                type="market",
                side=side,
                amount=amount,
            )
            order_id = order.get("id", "unknown")
            logger.info(
                f"[LIVE] {side.upper()} {proposal.symbol} amount={amount:.6f} "
                f"@ ~{price:.4f} order_id={order_id}"
            )

            # Place stop-loss / take-profit orders
            await self._place_sl_tp(proposal, amount, order_id)

            self.portfolio.add_position({
                "symbol": proposal.symbol,
                "id": order_id,
                "direction": proposal.direction,
            })
            return OrderResult(
                symbol=proposal.symbol, direction=proposal.direction,
                order_id=order_id, price=price, amount=amount, status="filled",
            )
        except Exception as exc:
            logger.error(f"Live order failed for {proposal.symbol}: {exc}")
            return OrderResult(
                symbol=proposal.symbol, direction=proposal.direction,
                order_id="error", price=price, amount=0.0, status="failed", error=str(exc),
            )

    async def _place_sl_tp(self, proposal: TradeProposal, amount: float, parent_id: str) -> None:
        """Attempt to place SL and TP bracket orders (exchange must support OCO)."""
        try:
            if proposal.direction == "long":
                await self._exchange.create_order(
                    symbol=proposal.symbol,
                    type="stop_market",
                    side="sell",
                    amount=amount,
                    params={"stopPrice": proposal.stop_loss},
                )
                await self._exchange.create_order(
                    symbol=proposal.symbol,
                    type="limit",
                    side="sell",
                    amount=amount,
                    price=proposal.take_profit,
                )
            else:
                await self._exchange.create_order(
                    symbol=proposal.symbol,
                    type="stop_market",
                    side="buy",
                    amount=amount,
                    params={"stopPrice": proposal.stop_loss},
                )
                await self._exchange.create_order(
                    symbol=proposal.symbol,
                    type="limit",
                    side="buy",
                    amount=amount,
                    price=proposal.take_profit,
                )
        except Exception as exc:
            logger.warning(f"SL/TP placement failed for {parent_id}: {exc} — manual management required")

    async def _connect_exchange(self) -> None:
        cfg = self.settings.exchange
        exchange_class = getattr(ccxt, cfg.id)
        params = {
            "apiKey": cfg.api_key,
            "secret": cfg.api_secret,
            "enableRateLimit": True,
        }
        if cfg.sandbox:
            params["sandbox"] = True
        self._exchange = exchange_class(params)
        logger.info(f"Connected to live exchange: {cfg.id}")

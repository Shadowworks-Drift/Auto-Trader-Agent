"""
Paper Trading Engine.

Simulates order execution, position management, stop-loss/take-profit
triggers, and P&L tracking without touching a real exchange.

State is persisted to SQLite so sessions survive restarts.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite
from loguru import logger


@dataclass
class PaperPosition:
    symbol: str
    direction: str          # "long" | "short"
    entry_price: float
    size_pct: float         # fraction of portfolio
    stop_loss: float
    take_profit: float
    opened_at: datetime
    entry_value: float      # absolute $ value at entry
    id: str = ""

    @property
    def size_units(self) -> float:
        return self.entry_value / max(self.entry_price, 1e-9)

    def unrealised_pnl(self, current_price: float) -> float:
        if self.direction == "long":
            return (current_price - self.entry_price) * self.size_units
        else:
            return (self.entry_price - current_price) * self.size_units

    def unrealised_pnl_pct(self, current_price: float) -> float:
        return self.unrealised_pnl(current_price) / max(self.entry_value, 1e-9)

    def should_stop_loss(self, price: float) -> bool:
        if self.direction == "long":
            return price <= self.stop_loss
        return price >= self.stop_loss

    def should_take_profit(self, price: float) -> bool:
        if self.direction == "long":
            return price >= self.take_profit
        return price <= self.take_profit


@dataclass
class PaperTrade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size_pct: float
    pnl: float
    pnl_pct: float
    exit_reason: str        # "stop_loss" | "take_profit" | "manual"
    opened_at: datetime
    closed_at: datetime


class PaperTradingEngine:
    """
    Full paper-trading simulation with SQLite persistence.

    Usage::

        engine = PaperTradingEngine(db_path="./data/trades.db", initial_balance=10_000)
        await engine.init()
        pos_id = await engine.open_position(proposal, current_price=42000.0)
        await engine.update_prices({"BTC/USDT": 43000.0})
        await engine.close()
    """

    def __init__(self, db_path: str = "./data/trades.db", initial_balance: float = 10_000.0) -> None:
        self.db_path = db_path
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, PaperPosition] = {}  # id → position
        self.trades: List[PaperTrade] = []
        self._db: Optional[aiosqlite.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        import os
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        await self._load_state()
        logger.info(
            f"PaperTradingEngine ready | balance=${self.balance:.2f} "
            f"| {len(self.positions)} open positions"
        )

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ── Trading operations ────────────────────────────────────────────────────

    async def open_position(self, proposal: Any, current_price: float) -> Optional[str]:
        """Open a paper position. Returns position ID or None if insufficient balance."""
        position_value = self.balance * proposal.position_size_pct
        if position_value <= 0:
            logger.warning(f"Position size too small for {proposal.symbol}")
            return None

        import uuid
        pos_id = str(uuid.uuid4())[:8]
        position = PaperPosition(
            id=pos_id,
            symbol=proposal.symbol,
            direction=proposal.direction,
            entry_price=current_price,
            size_pct=proposal.position_size_pct,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit,
            opened_at=datetime.utcnow(),
            entry_value=position_value,
        )
        self.positions[pos_id] = position
        self.balance -= position_value

        await self._save_position(pos_id, position)
        logger.info(
            f"[PAPER] OPEN {proposal.direction.upper()} {proposal.symbol} "
            f"@ {current_price:.4f} size=${position_value:.2f} "
            f"SL={proposal.stop_loss:.4f} TP={proposal.take_profit:.4f} [id={pos_id}]"
        )
        return pos_id

    async def update_prices(self, prices: Dict[str, float]) -> List[PaperTrade]:
        """Check all open positions against current prices, trigger SL/TP."""
        closed_trades: List[PaperTrade] = []
        for pos_id, pos in list(self.positions.items()):
            price = prices.get(pos.symbol)
            if price is None:
                continue
            reason = None
            if pos.should_stop_loss(price):
                reason = "stop_loss"
            elif pos.should_take_profit(price):
                reason = "take_profit"
            if reason:
                trade = await self._close_position(pos_id, price, reason)
                closed_trades.append(trade)
        return closed_trades

    async def close_position(self, pos_id: str, price: float, reason: str = "manual") -> Optional[PaperTrade]:
        if pos_id not in self.positions:
            return None
        return await self._close_position(pos_id, price, reason)

    # ── Portfolio metrics ─────────────────────────────────────────────────────

    def portfolio_value(self, prices: Dict[str, float]) -> float:
        """Total portfolio value = cash + unrealised P&L."""
        unrealised = sum(
            pos.entry_value + pos.unrealised_pnl(prices.get(pos.symbol, pos.entry_price))
            for pos in self.positions.values()
        )
        return self.balance + unrealised

    def total_return_pct(self, prices: Dict[str, float]) -> float:
        return (self.portfolio_value(prices) - self.initial_balance) / self.initial_balance

    def sharpe_ratio(self) -> float:
        """Annualised Sharpe from closed trades."""
        if len(self.trades) < 5:
            return 0.0
        import numpy as np
        returns = [t.pnl_pct for t in self.trades]
        mean = np.mean(returns)
        std = np.std(returns)
        if std < 1e-9:
            return 0.0
        trades_per_year = 252  # approximate for daily
        return float((mean / std) * (trades_per_year ** 0.5))

    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        equity = self.initial_balance
        peak = equity
        max_dd = 0.0
        for t in self.trades:
            equity += t.pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _close_position(self, pos_id: str, price: float, reason: str) -> PaperTrade:
        pos = self.positions.pop(pos_id)
        if pos.direction == "long":
            pnl = (price - pos.entry_price) * pos.size_units
        else:
            pnl = (pos.entry_price - price) * pos.size_units
        pnl_pct = pnl / max(pos.entry_value, 1e-9)
        proceeds = pos.entry_value + pnl
        self.balance += proceeds

        trade = PaperTrade(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=price,
            size_pct=pos.size_pct,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            opened_at=pos.opened_at,
            closed_at=datetime.utcnow(),
        )
        self.trades.append(trade)
        await self._save_trade(trade, pos_id)
        await self._remove_position_db(pos_id)

        emoji = "✓" if pnl > 0 else "✗"
        logger.info(
            f"[PAPER] {emoji} CLOSE {pos.direction.upper()} {pos.symbol} "
            f"@ {price:.4f} PnL={pnl:+.2f} ({pnl_pct:+.2%}) [{reason}]"
        )
        return trade

    # ── SQLite persistence ────────────────────────────────────────────────────

    async def _create_tables(self) -> None:
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                symbol TEXT, direction TEXT,
                entry_price REAL, size_pct REAL, stop_loss REAL, take_profit REAL,
                entry_value REAL, opened_at TEXT
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pos_id TEXT, symbol TEXT, direction TEXT,
                entry_price REAL, exit_price REAL, size_pct REAL,
                pnl REAL, pnl_pct REAL, exit_reason TEXT,
                opened_at TEXT, closed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS portfolio_state (
                key TEXT PRIMARY KEY, value REAL
            );
        """)
        await self._db.commit()

    async def _save_position(self, pos_id: str, pos: PaperPosition) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?,?)""",
            (pos_id, pos.symbol, pos.direction, pos.entry_price, pos.size_pct,
             pos.stop_loss, pos.take_profit, pos.entry_value, pos.opened_at.isoformat()),
        )
        await self._db.commit()

    async def _remove_position_db(self, pos_id: str) -> None:
        await self._db.execute("DELETE FROM positions WHERE id=?", (pos_id,))
        await self._db.commit()

    async def _save_trade(self, trade: PaperTrade, pos_id: str) -> None:
        await self._db.execute(
            """INSERT INTO trades
               (pos_id, symbol, direction, entry_price, exit_price, size_pct,
                pnl, pnl_pct, exit_reason, opened_at, closed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (pos_id, trade.symbol, trade.direction, trade.entry_price, trade.exit_price,
             trade.size_pct, trade.pnl, trade.pnl_pct, trade.exit_reason,
             trade.opened_at.isoformat(), trade.closed_at.isoformat()),
        )
        await self._db.commit()

    async def _load_state(self) -> None:
        async with self._db.execute("SELECT value FROM portfolio_state WHERE key='balance'") as cur:
            row = await cur.fetchone()
            if row:
                self.balance = float(row[0])
        async with self._db.execute("SELECT * FROM positions") as cur:
            rows = await cur.fetchall()
        for row in rows:
            pos = PaperPosition(
                id=row[0], symbol=row[1], direction=row[2],
                entry_price=float(row[3]), size_pct=float(row[4]),
                stop_loss=float(row[5]), take_profit=float(row[6]),
                entry_value=float(row[7]),
                opened_at=datetime.fromisoformat(row[8]),
            )
            self.positions[pos.id] = pos

    async def _persist_balance(self) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO portfolio_state VALUES ('balance', ?)", (self.balance,)
        )
        await self._db.commit()

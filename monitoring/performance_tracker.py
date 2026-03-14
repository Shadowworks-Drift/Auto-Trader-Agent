"""
Performance Tracker — collects metrics and exposes them via Prometheus.

Tracks:
  - Portfolio value / total return
  - Sharpe ratio
  - Max drawdown
  - Win rate
  - Per-symbol P&L
  - Agent decision latency
  - LLM inference latency
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics endpoint disabled")


class PerformanceTracker:
    """
    Tracks and exposes trading performance metrics.

    Usage::

        tracker = PerformanceTracker(metrics_port=8000)
        tracker.start_server()
        tracker.record_trade(symbol="BTC/USDT", direction="long", pnl=120.0, pnl_pct=0.012)
    """

    def __init__(self, metrics_port: int = 8000, enabled: bool = True) -> None:
        self.metrics_port = metrics_port
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        self._server_started = False
        self._trade_history: List[Dict[str, Any]] = []

        if self.enabled:
            self._setup_metrics()

    def _setup_metrics(self) -> None:
        self.portfolio_value = Gauge("trader_portfolio_value_usd", "Current portfolio value in USD")
        self.total_return = Gauge("trader_total_return_pct", "Total return percentage")
        self.sharpe_ratio = Gauge("trader_sharpe_ratio", "Annualised Sharpe ratio")
        self.max_drawdown = Gauge("trader_max_drawdown_pct", "Maximum drawdown percentage")
        self.win_rate = Gauge("trader_win_rate", "Win rate of closed trades")
        self.open_positions = Gauge("trader_open_positions", "Number of open positions")

        self.trades_total = Counter("trader_trades_total", "Total number of closed trades",
                                   ["symbol", "direction", "exit_reason"])
        self.trades_pnl = Gauge("trader_pnl_usd", "P&L per trade", ["symbol"])

        self.loop_duration = Histogram(
            "trader_loop_duration_seconds", "Main loop duration",
            buckets=[1, 5, 10, 30, 60, 120, 300],
        )
        self.llm_latency = Histogram(
            "trader_llm_latency_seconds", "LLM inference latency",
            ["agent"], buckets=[0.5, 1, 2, 5, 10, 30, 60, 90],
        )
        self.agent_confidence = Gauge("trader_agent_confidence", "Agent output confidence",
                                      ["agent", "symbol"])
        self.decisions_total = Counter("trader_decisions_total", "Total trade decisions",
                                      ["direction", "symbol"])

    def start_server(self) -> None:
        if not self.enabled or self._server_started:
            return
        try:
            start_http_server(self.metrics_port)
            self._server_started = True
            logger.info(f"Prometheus metrics server started on :{self.metrics_port}")
        except Exception as exc:
            logger.warning(f"Could not start metrics server: {exc}")

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_trade(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        pnl_pct: float,
        exit_reason: str = "unknown",
    ) -> None:
        entry = {
            "symbol": symbol,
            "direction": direction,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._trade_history.append(entry)
        if self.enabled:
            self.trades_total.labels(symbol=symbol, direction=direction, exit_reason=exit_reason).inc()
            self.trades_pnl.labels(symbol=symbol).set(pnl)

    def update_portfolio(self, value: float, return_pct: float, drawdown: float, open_pos: int) -> None:
        if self.enabled:
            self.portfolio_value.set(value)
            self.total_return.set(return_pct * 100)
            self.max_drawdown.set(drawdown * 100)
            self.open_positions.set(open_pos)

    def update_ratios(self, sharpe: float, win_rate: float) -> None:
        if self.enabled:
            self.sharpe_ratio.set(sharpe)
            self.win_rate.set(win_rate * 100)

    def record_llm_latency(self, agent: str, seconds: float) -> None:
        if self.enabled:
            self.llm_latency.labels(agent=agent).observe(seconds)

    def record_agent_confidence(self, agent: str, symbol: str, confidence: float) -> None:
        if self.enabled:
            self.agent_confidence.labels(agent=agent, symbol=symbol).set(confidence)

    def record_decision(self, direction: str, symbol: str) -> None:
        if self.enabled:
            self.decisions_total.labels(direction=direction, symbol=symbol).inc()

    def record_loop_duration(self, seconds: float) -> None:
        if self.enabled:
            self.loop_duration.observe(seconds)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        if not self._trade_history:
            return {"trades": 0}
        pnls = [t["pnl_pct"] for t in self._trade_history]
        wins = sum(1 for p in pnls if p > 0)
        import numpy as np
        return {
            "trades": len(self._trade_history),
            "win_rate": wins / len(pnls),
            "avg_pnl_pct": float(np.mean(pnls)),
            "total_pnl": sum(t["pnl"] for t in self._trade_history),
        }

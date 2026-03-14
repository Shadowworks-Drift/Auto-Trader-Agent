"""
Auto-Trader-Agent  ·  Main Entry Point
══════════════════════════════════════════════════════════════════════════════

Architecture (Adversarial Decision Framework):
  1. SymbolSelector   → screen universe → top-N candidates
  2. DataSync         → fetch multi-timeframe OHLCV + news
  3. QuantAnalyst     → technical indicators → numeric signal
  4. TrendAgent       → LLM macro-trend assessment
  5. SetupAgent       → LLM trade-setup quality evaluation
  6. TriggerAgent     → LLM precise entry trigger
  7. SentimentAgent   → LLM news / social sentiment
  8. DecisionCore     → weighted fusion + adversarial debate
  9. RiskAudit        → hard rule gates + LLM narrative review
  10. ExecutionEngine  → paper or live order placement
  11. Monitoring       → Prometheus metrics + Rich dashboard

Usage:
    python main.py                        # paper trading, default config
    python main.py --config config/my.yaml
    python main.py --mode live            # live trading (caution!)
    python main.py --symbols BTC/USDT ETH/USDT
    python main.py --once                 # run a single cycle and exit
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.decision_core import DecisionCore, TradeProposal
from agents.quant_analyst import QuantAnalyst
from agents.risk_audit import RiskAudit, PortfolioState
from agents.semantic_agents import TrendAgent, SetupAgent, TriggerAgent, SentimentAgent
from agents.symbol_selector import SymbolSelector
from config.settings import Settings, get_settings
from data.data_sync import DataSync
from data.market_data import MarketSnapshot
from execution.execution_engine import ExecutionEngine
from llm.ollama_client import OllamaClient
from monitoring.dashboard import Dashboard
from monitoring.performance_tracker import PerformanceTracker
from utils.logger import setup_logger

try:
    from rl.position_sizer import RLPositionSizer
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-Trader-Agent — Local LLM Multi-Agent Trading System"
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", choices=["paper", "live"], help="Override trading mode")
    parser.add_argument("--symbols", nargs="+", help="Override symbol list")
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM agents (quant-only mode)")
    parser.add_argument("--dashboard", action="store_true", default=True, help="Show Rich dashboard")
    return parser.parse_args()


# ── Main Orchestrator ──────────────────────────────────────────────────────────

class TradingOrchestrator:
    """
    Top-level controller that manages the full agent pipeline.
    Runs an async event loop with configurable interval.
    """

    def __init__(self, settings: Settings, use_llm: bool = True) -> None:
        self.settings = settings
        self.use_llm = use_llm
        self._running = False
        self._loop_count = 0
        self._errors = 0
        self._last_signals: List[Dict[str, Any]] = []

        # ── Component initialisation ─────────────────────────────────────
        self.llm = OllamaClient(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            fallback_model=settings.llm.fallback_model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=float(settings.llm.timeout_seconds),
        )
        self.data_sync = DataSync(settings)
        self.symbol_selector = SymbolSelector(settings, self.data_sync)
        self.quant_analyst = QuantAnalyst(settings)

        # ── Optional RL position sizer ────────────────────────────────────
        self.rl_sizer = None
        if RL_AVAILABLE:
            self.rl_sizer = RLPositionSizer.from_model(max_position_size=settings.trading.position_size_pct)

        if use_llm:
            self.trend_agent = TrendAgent(settings, self.llm)
            self.setup_agent = SetupAgent(settings, self.llm)
            self.trigger_agent = TriggerAgent(settings, self.llm)
            self.sentiment_agent = SentimentAgent(settings, self.llm)
            self.decision_core = DecisionCore(settings, self.llm, rl_sizer=self.rl_sizer)
        else:
            self.trend_agent = None
            self.setup_agent = None
            self.trigger_agent = None
            self.sentiment_agent = None
            self.decision_core = None

        self.execution = ExecutionEngine(settings)
        self.risk_audit = RiskAudit(settings, self.execution.portfolio, self.llm if use_llm else None)
        self.tracker = PerformanceTracker(
            metrics_port=settings.monitoring.metrics_port,
            enabled=settings.monitoring.metrics_enabled,
        )
        self.dashboard = Dashboard(refresh_seconds=settings.monitoring.dashboard_refresh_seconds)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info("Starting Auto-Trader-Agent orchestrator...")
        await self.data_sync.connect()
        await self.execution.init()
        self.tracker.start_server()
        self.dashboard.print_banner()

        if self.use_llm:
            healthy = await self.llm.health_check()
            if not healthy:
                logger.warning(
                    f"Ollama server not reachable at {self.settings.llm.base_url}. "
                    "LLM agents will fail — switch to --no-llm mode or start Ollama."
                )

        self._running = True
        logger.info(
            f"Mode: {self.settings.trading.mode.upper()} | "
            f"Symbols: {self.settings.trading.symbols} | "
            f"LLM: {self.settings.llm.model}"
        )

    async def stop(self) -> None:
        logger.info("Shutting down orchestrator...")
        self._running = False
        await self.data_sync.disconnect()
        await self.execution.shutdown()

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run_loop(self, once: bool = False) -> None:
        """Main trading loop."""
        interval = self.settings.trading.loop_interval_seconds
        while self._running:
            loop_start = time.perf_counter()
            try:
                await self._cycle()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                self._errors += 1
                logger.error(f"Loop error: {exc}", exc_info=True)

            elapsed = time.perf_counter() - loop_start
            self.tracker.record_loop_duration(elapsed)

            if once:
                break

            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                logger.debug(f"Sleeping {sleep_time:.0f}s until next cycle...")
                await asyncio.sleep(sleep_time)

    # ── Trading cycle ─────────────────────────────────────────────────────────

    async def _cycle(self) -> None:
        self._loop_count += 1
        logger.info(f"═══ Cycle #{self._loop_count} — {datetime.utcnow().strftime('%H:%M:%S UTC')} ═══")

        symbols = self.settings.trading.symbols

        # ── 1. Symbol selection ────────────────────────────────────────────
        try:
            selected_symbols = await self.symbol_selector.select(symbols)
        except Exception as exc:
            logger.warning(f"Symbol selection failed, using full list: {exc}")
            selected_symbols = symbols

        if not selected_symbols:
            logger.warning("No symbols selected after filtering — skipping cycle")
            return

        # ── 2. Fetch data for all symbols concurrently ─────────────────────
        snapshots = await self.data_sync.fetch_all_snapshots(selected_symbols)

        prices = {sym: snap.current_price for sym, snap in snapshots.items()}

        # ── 3. Update open positions with current prices ───────────────────
        closed_trades = await self.execution.update_positions(prices)
        for trade in closed_trades:
            self.tracker.record_trade(
                symbol=trade.symbol,
                direction=trade.direction,
                pnl=trade.pnl,
                pnl_pct=trade.pnl_pct,
                exit_reason=trade.exit_reason,
            )

        # ── 4. Analyse each symbol ─────────────────────────────────────────
        cycle_signals: List[Dict[str, Any]] = []
        for symbol, snapshot in snapshots.items():
            try:
                signal = await self._analyse_symbol(snapshot)
                if signal:
                    cycle_signals.append(signal)
            except Exception as exc:
                logger.error(f"Analysis failed for {symbol}: {exc}", exc_info=True)
                self._errors += 1

        self._last_signals = cycle_signals

        # ── 5. Update dashboard & metrics ─────────────────────────────────
        perf = self.execution.get_performance_summary(prices)
        self.tracker.update_portfolio(
            value=perf.get("portfolio_value", 0),
            return_pct=perf.get("total_return_pct", 0),
            drawdown=perf.get("max_drawdown", 0),
            open_pos=perf.get("open_positions", 0),
        )
        self.tracker.update_ratios(
            sharpe=perf.get("sharpe_ratio", 0),
            win_rate=perf.get("win_rate", 0),
        )
        self.dashboard.update(self._build_state(perf, prices))

    async def _analyse_symbol(self, snapshot: MarketSnapshot) -> Optional[Dict[str, Any]]:
        symbol = snapshot.symbol

        # ── Quant analysis (always runs) ───────────────────────────────────
        quant_result = await self.quant_analyst.analyse(snapshot)
        if not quant_result.success:
            logger.warning(f"Quant failed for {symbol}: {quant_result.error}")
            return None

        quant_summary = quant_result.data.get("summary_text", "")
        self.tracker.record_agent_confidence("quant", symbol, quant_result.confidence)

        # ── LLM agents (when enabled) ──────────────────────────────────────
        if self.use_llm and all([self.trend_agent, self.setup_agent, self.trigger_agent]):
            trend_result = await self.trend_agent.analyse(snapshot, quant_summary)
            setup_result = await self.setup_agent.analyse(snapshot, trend_result, quant_summary)
            trigger_result = await self.trigger_agent.analyse(snapshot, setup_result)
            sentiment_result = await self.sentiment_agent.analyse(snapshot)

            for agent_name, result in [
                ("trend", trend_result), ("setup", setup_result),
                ("trigger", trigger_result), ("sentiment", sentiment_result)
            ]:
                self.tracker.record_agent_confidence(agent_name, symbol, result.confidence)

            # ── Decision fusion ────────────────────────────────────────────
            proposal = await self.decision_core.decide(
                snapshot, quant_result, trend_result, setup_result, trigger_result, sentiment_result
            )
        else:
            # Quant-only fallback
            proposal = _quant_only_proposal(snapshot, quant_result, self.settings)

        self.tracker.record_decision(proposal.direction, symbol)

        # ── Risk audit ─────────────────────────────────────────────────────
        if proposal.is_actionable(self.settings.decision.min_confidence):
            agent_signals_text = quant_summary
            risk_decision = await self.risk_audit.audit(proposal, agent_signals_text)
            self.dashboard.print_decision(proposal, risk_decision)

            # ── Execute ────────────────────────────────────────────────────
            result = await self.execution.execute(proposal, risk_decision, snapshot.current_price)
            logger.info(
                f"Order [{symbol}]: status={result.status} order_id={result.order_id}"
            )
        else:
            logger.info(
                f"[{symbol}] No trade — {proposal.direction} confidence={proposal.confidence:.2f} "
                f"< min={self.settings.decision.min_confidence}"
            )

        return {
            "symbol": symbol,
            "direction": proposal.direction,
            "confidence": proposal.confidence,
        }

    def _build_state(self, perf: Dict[str, Any], prices: Dict[str, float]) -> Dict[str, Any]:
        return {
            "mode": self.settings.trading.mode,
            "llm_model": self.settings.llm.model,
            "performance": perf,
            "positions": self.execution.get_open_positions(),
            "last_signals": self._last_signals,
            "loop_count": self._loop_count,
            "last_loop_at": datetime.utcnow().strftime("%H:%M:%S"),
            "errors": self._errors,
        }


# ── Quant-only fallback proposal ──────────────────────────────────────────────

def _quant_only_proposal(
    snapshot: MarketSnapshot,
    quant_result: Any,
    settings: Settings,
) -> "TradeProposal":
    from agents.decision_core import TradeProposal
    direction = quant_result.data.get("direction", "none")
    score = quant_result.data.get("score", 0.0)
    price = snapshot.current_price
    cfg = settings.risk
    if direction == "long":
        sl = price * (1 - cfg.stop_loss_pct)
        tp = price * (1 + cfg.take_profit_pct)
    elif direction == "short":
        sl = price * (1 + cfg.stop_loss_pct)
        tp = price * (1 - cfg.take_profit_pct)
    else:
        sl = tp = price

    return TradeProposal(
        symbol=snapshot.symbol,
        direction=direction,
        confidence=float(quant_result.confidence),
        entry_price=price,
        stop_loss=sl,
        take_profit=tp,
        position_size_pct=settings.trading.position_size_pct,
        reasoning=quant_result.data.get("summary_text", "Quant-only mode"),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

async def async_main(args: argparse.Namespace) -> None:
    # Load settings
    cfg_path = Path(args.config)
    settings = Settings.from_yaml(cfg_path)

    # CLI overrides
    if args.mode:
        settings.trading.mode = args.mode
    if args.symbols:
        settings.trading.symbols = args.symbols

    # Logger
    setup_logger(settings.monitoring.log_level, settings.monitoring.log_file)

    orchestrator = TradingOrchestrator(settings, use_llm=not args.no_llm)

    # Graceful shutdown on SIGTERM/SIGINT
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(orchestrator.stop()))

    await orchestrator.start()
    try:
        await orchestrator.run_loop(once=args.once)
    finally:
        await orchestrator.stop()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        print(f"Fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

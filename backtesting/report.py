"""
Backtest Report Generator
══════════════════════════
Produces a comprehensive Rich terminal report and optional HTML export.

Metrics (following Two Sigma / industry standard):
  - Total return, CAGR
  - Sharpe, Sortino, Calmar ratios
  - Max drawdown, avg drawdown, drawdown duration
  - Win rate, profit factor, avg win/loss, expectancy
  - Per-regime breakdown
  - Walk-forward consistency score
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .backtest_engine import BacktestResults, BacktestTrade


class BacktestReport:
    """
    Computes and displays backtest performance statistics.

    Usage::

        report = BacktestReport(results)
        report.print()
        report.save_json("results/backtest.json")
    """

    def __init__(self, results: BacktestResults) -> None:
        self.results = results
        self._stats: Optional[Dict[str, Any]] = None

    # ── Core statistics ───────────────────────────────────────────────────────

    def compute_stats(self) -> Dict[str, Any]:
        if self._stats is not None:
            return self._stats

        trades = self.results.trades
        equity = self.results.equity_curve
        cfg    = self.results.config

        if not trades or equity.empty:
            self._stats = {"error": "No trades to analyse"}
            return self._stats

        initial = cfg.initial_capital
        final   = float(equity.iloc[-1])
        total_return = (final - initial) / initial

        # Duration
        duration_days = (equity.index[-1] - equity.index[0]).days if hasattr(equity.index[0], 'year') else len(equity)
        years = max(duration_days / 365.25, 1e-3)
        cagr  = (final / initial) ** (1 / years) - 1

        # Per-trade stats
        pnl_pcts = [t.pnl_after_costs / cfg.initial_capital for t in trades]
        wins  = [t for t in trades if t.pnl_after_costs > 0]
        losses = [t for t in trades if t.pnl_after_costs <= 0]
        win_rate = len(wins) / len(trades)

        avg_win  = np.mean([t.pnl_after_costs for t in wins])   if wins   else 0.0
        avg_loss = np.mean([t.pnl_after_costs for t in losses]) if losses else 0.0
        profit_factor = abs(sum(t.pnl_after_costs for t in wins)) / max(abs(sum(t.pnl_after_costs for t in losses)), 1e-9)
        expectancy    = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Ratio metrics (daily returns from equity curve)
        daily_returns = equity.pct_change().dropna()
        sharpe  = _sharpe(daily_returns)
        sortino = _sortino(daily_returns)

        # Drawdown
        dd_series = self.results.drawdown_series
        max_dd    = float(dd_series.min()) if not dd_series.empty else 0.0
        avg_dd    = float(dd_series.mean()) if not dd_series.empty else 0.0
        calmar    = cagr / abs(max_dd) if max_dd != 0 else 0.0

        # Costs
        total_costs = sum(t.costs for t in trades)
        costs_pct   = total_costs / initial

        # Exit reason breakdown
        reasons: Dict[str, int] = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        # Walk-forward consistency
        wf_windows = self.results.walk_forward_windows
        if wf_windows:
            wf_returns = [w["return_pct"] for w in wf_windows]
            wf_positive = sum(1 for r in wf_returns if r > 0)
            wf_consistency = wf_positive / len(wf_windows)
        else:
            wf_consistency = None

        self._stats = {
            "symbol": self.results.symbol,
            "total_trades": len(trades),
            "initial_capital": initial,
            "final_capital": final,
            "total_return_pct": total_return * 100,
            "cagr_pct": cagr * 100,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown_pct": max_dd * 100,
            "avg_drawdown_pct": avg_dd * 100,
            "win_rate_pct": win_rate * 100,
            "profit_factor": profit_factor,
            "expectancy_usd": expectancy,
            "avg_win_usd": avg_win,
            "avg_loss_usd": avg_loss,
            "total_costs_usd": total_costs,
            "costs_pct": costs_pct * 100,
            "exit_reasons": reasons,
            "wf_consistency_pct": wf_consistency * 100 if wf_consistency is not None else None,
            "duration_days": duration_days,
        }
        return self._stats

    # ── Rich terminal output ──────────────────────────────────────────────────

    def print(self, console: Optional[Console] = None) -> None:
        console = console or Console()
        s = self.compute_stats()

        if "error" in s:
            console.print(f"[red]Backtest Error: {s['error']}[/red]")
            return

        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Backtest Report — {s['symbol']}[/bold cyan]\n"
            f"[dim]{s['duration_days']} days | {s['total_trades']} trades[/dim]",
            border_style="cyan"
        ))

        # ── Summary table ──────────────────────────────────────────────────
        t1 = Table(title="Performance Summary", box=box.ROUNDED, show_header=True, header_style="bold cyan")
        t1.add_column("Metric")
        t1.add_column("Value", justify="right")

        def _colour(v, good_pos=True):
            colour = "green" if (v > 0) == good_pos else "red"
            return f"[{colour}]{v:.2f}[/{colour}]"

        t1.add_row("Total Return", f"[{'green' if s['total_return_pct'] > 0 else 'red'}]{s['total_return_pct']:+.2f}%[/]")
        t1.add_row("CAGR", f"[{'green' if s['cagr_pct'] > 0 else 'red'}]{s['cagr_pct']:+.2f}%[/]")
        t1.add_row("Sharpe Ratio", _colour(s["sharpe"]))
        t1.add_row("Sortino Ratio", _colour(s["sortino"]))
        t1.add_row("Calmar Ratio", _colour(s["calmar"]))
        t1.add_row("Max Drawdown", f"[red]{s['max_drawdown_pct']:.2f}%[/red]")
        t1.add_row("Win Rate", f"{s['win_rate_pct']:.1f}%")
        pf = s["profit_factor"]
        t1.add_row("Profit Factor", f"[{'green' if pf >= 1 else 'red'}]{pf:.2f}[/]")
        t1.add_row("Expectancy", f"${s['expectancy_usd']:+.2f}")
        t1.add_row("Total Costs", f"[yellow]${s['total_costs_usd']:.2f} ({s['costs_pct']:.2f}%)[/yellow]")
        if s["wf_consistency_pct"] is not None:
            t1.add_row("WF Consistency", f"{s['wf_consistency_pct']:.1f}% windows profitable")
        console.print(t1)

        # ── Exit reason breakdown ──────────────────────────────────────────
        t2 = Table(title="Exit Reasons", box=box.SIMPLE)
        t2.add_column("Reason")
        t2.add_column("Count", justify="right")
        t2.add_column("% of Trades", justify="right")
        total = s["total_trades"]
        for reason, count in sorted(s["exit_reasons"].items(), key=lambda x: -x[1]):
            t2.add_row(reason, str(count), f"{count / total * 100:.1f}%")
        console.print(t2)

        # ── Walk-forward windows ───────────────────────────────────────────
        wf_windows = self.results.walk_forward_windows
        if wf_windows:
            t3 = Table(title="Walk-Forward Windows", box=box.SIMPLE)
            t3.add_column("Test Period")
            t3.add_column("Trades", justify="right")
            t3.add_column("Return", justify="right")
            for w in wf_windows:
                ret = w["return_pct"] * 100
                colour = "green" if ret > 0 else "red"
                t3.add_row(
                    f"{w['test_start']:%Y-%m-%d} → {w['test_end']:%Y-%m-%d}",
                    str(w["trades"]),
                    f"[{colour}]{ret:+.2f}%[/{colour}]",
                )
            console.print(t3)

        # ── Verdict ────────────────────────────────────────────────────────
        verdict = self._verdict(s)
        console.print(Panel(verdict, title="Strategy Verdict", border_style="yellow"))

    def _verdict(self, s: Dict[str, Any]) -> str:
        issues: List[str] = []
        positives: List[str] = []

        if s["sharpe"] >= 1.5:
            positives.append(f"Strong Sharpe ({s['sharpe']:.2f})")
        elif s["sharpe"] < 0.5:
            issues.append(f"Weak Sharpe ({s['sharpe']:.2f}) — not enough risk-adjusted return")

        if s["win_rate_pct"] >= 55:
            positives.append(f"Good win rate ({s['win_rate_pct']:.1f}%)")
        elif s["win_rate_pct"] < 40:
            issues.append(f"Low win rate ({s['win_rate_pct']:.1f}%) — check R:R ratio")

        if s["max_drawdown_pct"] > -25:
            positives.append(f"Acceptable drawdown ({s['max_drawdown_pct']:.1f}%)")
        else:
            issues.append(f"Large drawdown ({s['max_drawdown_pct']:.1f}%) — review risk management")

        if s["profit_factor"] >= 1.5:
            positives.append(f"Good profit factor ({s['profit_factor']:.2f})")
        elif s["profit_factor"] < 1.0:
            issues.append("Profit factor < 1.0 — strategy loses money after costs")

        if s["wf_consistency_pct"] is not None:
            if s["wf_consistency_pct"] >= 60:
                positives.append(f"Walk-forward consistent ({s['wf_consistency_pct']:.0f}% windows profitable)")
            else:
                issues.append(f"Poor walk-forward consistency ({s['wf_consistency_pct']:.0f}%) — likely overfitting")

        lines = []
        if positives:
            lines.append("[green]Strengths:[/green]")
            lines.extend(f"  ✓ {p}" for p in positives)
        if issues:
            lines.append("[red]Concerns:[/red]")
            lines.extend(f"  ✗ {i}" for i in issues)

        if not issues and positives:
            lines.append("\n[bold green]→ READY for extended paper trading[/bold green]")
        elif len(issues) <= 1:
            lines.append("\n[bold yellow]→ MARGINAL — address concerns before paper trading[/bold yellow]")
        else:
            lines.append("\n[bold red]→ NOT READY — significant issues detected[/bold red]")

        return "\n".join(lines)

    # ── Export ────────────────────────────────────────────────────────────────

    def save_json(self, path: str) -> None:
        stats = self.compute_stats()
        # Convert non-serialisable types
        clean = {k: (v.isoformat() if hasattr(v, 'isoformat') else v) for k, v in stats.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(clean, f, indent=2, default=str)
        logger.info(f"Backtest results saved to {path}")

    def trade_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        if not self.results.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.results.trades])


# ── Statistical helpers ───────────────────────────────────────────────────────

def _sharpe(daily_returns: "pd.Series", risk_free: float = 0.05 / 252) -> float:
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0
    excess = daily_returns - risk_free
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def _sortino(daily_returns: "pd.Series", risk_free: float = 0.05 / 252) -> float:
    if daily_returns.empty:
        return 0.0
    excess = daily_returns - risk_free
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float((excess.mean() / downside) * np.sqrt(252))

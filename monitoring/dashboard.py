"""
Auto-Trader-Agent  ·  Rich Live Terminal Dashboard
════════════════════════════════════════════════════════════════════════════════

Full-screen layout:
  ┌─ header ──────────────────────────────────────────────────────────────────┐
  ├─ portfolio ──┬─ agent scores ──────────────────┬─ market prices ──────────┤
  ├─ open positions (with live PnL) ───────────────────────────────────────────┤
  ├─ decision log (scrolling, last N entries) ─────────────────────────────────┤
  ├─ trade history (last N closed trades) ─────────────────────────────────────┤
  └─ footer (countdown · errors · uptime) ────────────────────────────────────┘
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ── Constants ─────────────────────────────────────────────────────────────────

_AGENT_ORDER = ["quant", "trend", "setup", "trigger", "sentiment", "vision", "regime"]
_AGENT_COLOURS = {
    "quant":     "cyan",
    "trend":     "blue",
    "setup":     "magenta",
    "trigger":   "yellow",
    "sentiment": "green",
    "vision":    "bright_cyan",
    "regime":    "bright_white",
}
_BAR_WIDTH = 10  # characters for confidence bar


def _bar(value: float, width: int = _BAR_WIDTH) -> Text:
    """Return a coloured Unicode block-char progress bar for a 0-1 value."""
    filled = max(0, min(width, round(value * width)))
    empty  = width - filled
    pct    = value * 100
    if pct >= 70:
        colour = "bold green"
    elif pct >= 45:
        colour = "yellow"
    else:
        colour = "red"
    bar_str = "█" * filled + "░" * empty
    return Text(bar_str, style=colour)


def _dir_style(direction: str) -> str:
    return {"long": "bold green", "short": "bold red"}.get(direction, "dim")


def _pnl_style(val: float) -> str:
    return "bold green" if val >= 0 else "bold red"


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ── Dashboard ──────────────────────────────────────────────────────────────────

class Dashboard:
    """
    Rich full-screen Live dashboard.

    Usage::

        dashboard = Dashboard(refresh_seconds=10)
        with dashboard.live():
            # inside trading loop:
            dashboard.update(state_dict)
    """

    def __init__(self, refresh_seconds: float = 10.0) -> None:
        self.refresh_seconds = refresh_seconds
        self.console = Console()
        self._live: Optional[Live] = None
        self._state: Dict[str, Any] = {}
        self._start_time: float = time.monotonic()

    # ── Context manager ────────────────────────────────────────────────────────

    @contextmanager
    def live(self):  # type: ignore[override]
        self._start_time = time.monotonic()
        with Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,        # always smooth; we throttle updates in main loop
            screen=True,
            vertical_overflow="visible",
        ) as live:
            self._live = live
            try:
                yield self
            finally:
                self._live = None

    def update(self, state: Dict[str, Any]) -> None:
        """Push a new state dict and refresh the display."""
        self._state = state
        if self._live:
            self._live.update(self._render())

    # ── Legacy helpers (still used by main.py print_decision / print_banner) ──

    def print_banner(self) -> None:
        """Printed once before Live starts — shows a compact startup banner."""
        self.console.print(
            Panel.fit(
                "[bold cyan]Auto-Trader-Agent[/bold cyan] · Local LLM Trading System\n"
                "[dim]Powered by Ollama · Multi-Agent Adversarial Decision Framework[/dim]",
                border_style="cyan",
            )
        )

    def print_decision(self, proposal: Any, risk: Any) -> None:
        """Called from main.py on each actionable signal — appended to decision log."""
        # The dashboard state is updated via update() each cycle; nothing printed here
        # when Live is active.  When running without Live (once mode), still print.
        if self._live:
            return
        if not risk.approved:
            self.console.print(f"[red]VETOED[/red] {proposal.symbol} — {risk.veto_reason}")
            return
        colour = "green" if proposal.direction == "long" else "red"
        self.console.print(
            f"[{colour}]▶ {proposal.direction.upper()}[/{colour}] "
            f"[bold]{proposal.symbol}[/bold] "
            f"conf=[yellow]{proposal.confidence:.0%}[/yellow] "
            f"entry={proposal.entry_price:.4f}  "
            f"SL={proposal.stop_loss:.4f}  TP={proposal.take_profit:.4f}  "
            f"R:R={proposal.risk_reward:.1f}"
        )

    # ── Top-level render ───────────────────────────────────────────────────────

    def _render(self) -> Layout:
        s = self._state
        layout = Layout()
        layout.split_column(
            Layout(name="header",    size=3),
            Layout(name="mid",       size=10),
            Layout(name="positions", size=8),
            Layout(name="decisions", size=10),
            Layout(name="history",   size=8),
            Layout(name="footer",    size=3),
        )

        layout["header"].update(self._header(s))

        layout["mid"].split_row(
            Layout(self._portfolio_panel(s),    name="portfolio", ratio=3),
            Layout(self._agent_scores_panel(s), name="agents",    ratio=4),
            Layout(self._prices_panel(s),       name="prices",    ratio=3),
        )

        layout["positions"].update(self._positions_panel(s))
        layout["decisions"].update(self._decisions_panel(s))
        layout["history"].update(self._history_panel(s))
        layout["footer"].update(self._footer(s))

        return layout

    # ── Section renderers ──────────────────────────────────────────────────────

    def _header(self, s: Dict[str, Any]) -> Panel:
        mode  = s.get("mode", "paper").upper()
        model = s.get("llm_model", "—")
        now   = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S  UTC")
        cycle = s.get("loop_count", 0)
        vision_model = s.get("vision_model", "")

        mode_colour = "bold green" if mode == "LIVE" else "bold yellow"
        model_tag   = f"[cyan]{model}[/cyan]"
        if vision_model:
            model_tag += f"  [bright_cyan]+{vision_model}[/bright_cyan]"

        content = (
            f"[{mode_colour}]  {mode}  [/{mode_colour}]"
            f"  │  LLM: {model_tag}"
            f"  │  [dim]{now}[/dim]"
            f"  │  Cycle [bold]#{cycle}[/bold]"
        )
        return Panel(Align.center(content), style="on grey11", border_style="bright_blue")

    def _portfolio_panel(self, s: Dict[str, Any]) -> Panel:
        perf  = s.get("performance", {})
        value = perf.get("portfolio_value", 0.0)
        ret   = perf.get("total_return_pct", 0.0)
        sharpe = perf.get("sharpe_ratio", 0.0)
        dd    = perf.get("max_drawdown", 0.0)
        wr    = perf.get("win_rate", 0.0)
        trades = int(perf.get("total_trades", 0))
        open_n = int(perf.get("open_positions", 0))

        ret_c = _pnl_style(ret)
        dd_c  = "bold red" if dd < -0.05 else "yellow" if dd < -0.02 else "green"
        wr_c  = "bold green" if wr >= 0.55 else "yellow" if wr >= 0.45 else "red"

        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="dim", width=12)
        grid.add_column(justify="right", width=14)
        grid.add_row("Value",    f"[bold white]${value:>12,.2f}[/bold white]")
        grid.add_row("Return",   Text(f"{ret:>+.2%}", style=ret_c))
        grid.add_row("Sharpe",   f"[bold]{sharpe:>+.3f}[/bold]")
        grid.add_row("Max DD",   Text(f"{dd:>.2%}", style=dd_c))
        grid.add_row("Win Rate", Text(f"{wr:.1%}  ({trades} trades)", style=wr_c))
        grid.add_row("Open",     f"[bold]{open_n}[/bold] position{'s' if open_n != 1 else ''}")

        return Panel(grid, title="[bold blue]Portfolio[/bold blue]", border_style="blue", padding=(0, 1))

    def _agent_scores_panel(self, s: Dict[str, Any]) -> Panel:
        scores: Dict[str, float] = s.get("agent_scores", {})
        symbol: str = s.get("latest_symbol", "")
        direction: str = s.get("latest_direction", "none")
        regime: str    = s.get("latest_regime", "")

        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="dim",       width=12)
        grid.add_column(width=_BAR_WIDTH + 1)
        grid.add_column(justify="right",   width=6)

        for agent in _AGENT_ORDER:
            if agent not in scores:
                continue
            val    = scores[agent]
            colour = _AGENT_COLOURS.get(agent, "white")
            label  = Text(f"{agent:<10}", style=colour)
            bar    = _bar(val)
            pct    = Text(f"{val:.0%}", style="bold")
            grid.add_row(label, bar, pct)

        # Direction summary row
        if direction != "none" and symbol:
            style = _dir_style(direction)
            grid.add_row(
                Text("─" * 10, style="dim"),
                Text("─" * (_BAR_WIDTH + 1), style="dim"),
                Text("─" * 6, style="dim"),
            )
            grid.add_row(
                Text(f"{symbol}", style="bold white"),
                Text(f"▶ {direction.upper()}", style=style),
                Text(regime[:6] if regime else "", style="dim"),
            )

        title = "[bold magenta]Agent Scores[/bold magenta]"
        if symbol:
            title += f"  [dim]{symbol}[/dim]"
        return Panel(grid, title=title, border_style="magenta", padding=(0, 1))

    def _prices_panel(self, s: Dict[str, Any]) -> Panel:
        prices: Dict[str, float] = s.get("prices", {})
        prev_prices: Dict[str, float] = s.get("prev_prices", {})

        if not prices:
            return Panel(
                Align.center("[dim]Fetching prices…[/dim]", vertical="middle"),
                title="[bold green]Market Prices[/bold green]",
                border_style="green",
            )

        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column(style="cyan bold", width=12)
        table.add_column(justify="right", width=12)
        table.add_column(justify="right", width=7)

        for symbol, price in list(prices.items())[:8]:
            prev  = prev_prices.get(symbol, price)
            delta = (price - prev) / max(prev, 1e-9)
            arr   = "▲" if delta > 0.0001 else "▼" if delta < -0.0001 else " "
            c     = "green" if delta > 0.0001 else "red" if delta < -0.0001 else "dim"
            # Format price smartly
            if price >= 1000:
                p_str = f"${price:>10,.0f}"
            elif price >= 1:
                p_str = f"${price:>10,.2f}"
            else:
                p_str = f"${price:>10,.4f}"
            table.add_row(
                symbol.replace("/USDT", ""),
                p_str,
                Text(f"{arr} {abs(delta):.2%}", style=c),
            )

        return Panel(table, title="[bold green]Market Prices[/bold green]", border_style="green", padding=(0, 1))

    def _positions_panel(self, s: Dict[str, Any]) -> Panel:
        positions: Dict[str, Any] = s.get("positions", {})
        prices: Dict[str, float]  = s.get("prices", {})

        if not positions:
            return Panel(
                Align.center("[dim]No open positions[/dim]", vertical="middle"),
                title="[bold yellow]Open Positions[/bold yellow]",
                border_style="yellow",
            )

        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold dim",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Symbol",   style="cyan bold",  width=12)
        table.add_column("Dir",      width=6)
        table.add_column("Entry",    justify="right",    width=12)
        table.add_column("Current",  justify="right",    width=12)
        table.add_column("SL",       justify="right",    width=12)
        table.add_column("TP",       justify="right",    width=12)
        table.add_column("Unreal PnL", justify="right",  width=14)
        table.add_column("%",        justify="right",    width=8)
        table.add_column("Size %",   justify="right",    width=8)
        table.add_column("Opened",   justify="right",    width=10)

        for pos_id, pos in list(positions.items())[:5]:
            sym   = pos.get("symbol", "?")
            dirn  = pos.get("direction", "?")
            ep    = float(pos.get("entry_price", 0))
            sl    = float(pos.get("stop_loss", 0))
            tp    = float(pos.get("take_profit", 0))
            ev    = float(pos.get("entry_value", 0))
            sp    = float(pos.get("size_pct", 0))
            cur   = prices.get(sym, ep)
            units = ev / max(ep, 1e-9)

            if dirn == "long":
                upnl     = (cur - ep) * units
                upnl_pct = (cur - ep) / max(ep, 1e-9)
            else:
                upnl     = (ep - cur) * units
                upnl_pct = (ep - cur) / max(ep, 1e-9)

            opened_at = pos.get("opened_at", "")
            if hasattr(opened_at, "strftime"):
                opened_str = opened_at.strftime("%H:%M")
            elif isinstance(opened_at, str) and "T" in opened_at:
                opened_str = opened_at[11:16]
            else:
                opened_str = str(opened_at)[:5]

            dir_txt = Text(dirn.upper(), style=_dir_style(dirn))
            pnl_txt = Text(f"${upnl:+,.2f}",  style=_pnl_style(upnl))
            pct_txt = Text(f"{upnl_pct:+.2%}", style=_pnl_style(upnl_pct))

            def _fp(v: float) -> str:
                return f"${v:,.0f}" if v >= 1000 else f"${v:,.2f}"

            table.add_row(
                sym,
                dir_txt,
                _fp(ep),
                _fp(cur),
                _fp(sl),
                _fp(tp),
                pnl_txt,
                pct_txt,
                f"{sp:.1%}",
                opened_str,
            )

        return Panel(table, title="[bold yellow]Open Positions[/bold yellow]", border_style="yellow", padding=(0, 1))

    def _decisions_panel(self, s: Dict[str, Any]) -> Panel:
        decisions: List[Dict[str, Any]] = s.get("recent_decisions", [])

        if not decisions:
            return Panel(
                Align.center("[dim]Waiting for first analysis cycle…[/dim]", vertical="middle"),
                title="[bold magenta]Decision Log[/bold magenta]",
                border_style="magenta",
            )

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold dim",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Time",      width=8,  style="dim")
        table.add_column("Symbol",    width=12, style="cyan bold")
        table.add_column("Decision",  width=8)
        table.add_column("Conf",      width=6,  justify="right")
        table.add_column("Bar",       width=_BAR_WIDTH + 1)
        table.add_column("Regime",    width=16, style="dim")
        table.add_column("Patterns",  ratio=1)
        table.add_column("Executed",  width=10)

        for dec in reversed(decisions[-8:]):
            dirn     = dec.get("direction", "none")
            conf     = float(dec.get("confidence", 0))
            sym      = dec.get("symbol", "?")
            ts       = dec.get("timestamp", "")
            regime   = dec.get("regime", "")
            patterns = dec.get("patterns", [])
            executed = dec.get("executed", "")

            if hasattr(ts, "strftime"):
                ts_str = ts.strftime("%H:%M:%S")
            elif isinstance(ts, str) and len(ts) >= 8:
                ts_str = ts[11:19] if "T" in ts else ts[:8]
            else:
                ts_str = str(ts)[:8]

            dir_style = _dir_style(dirn)
            arrow = "▶" if dirn == "long" else "◀" if dirn == "short" else "—"
            dir_txt = Text(f"{arrow} {dirn.upper()}", style=dir_style)

            pat_str = ", ".join(patterns[:3]) if patterns else "[dim]—[/dim]"

            exec_colour = "green" if executed == "filled" else "red" if executed == "vetoed" else "dim"
            exec_txt = Text(executed or "—", style=exec_colour)

            table.add_row(
                ts_str,
                sym,
                dir_txt,
                Text(f"{conf:.0%}", style="bold"),
                _bar(conf),
                regime[:16] if regime else "—",
                pat_str,
                exec_txt,
            )

        return Panel(table, title="[bold magenta]Decision Log[/bold magenta]", border_style="magenta", padding=(0, 1))

    def _history_panel(self, s: Dict[str, Any]) -> Panel:
        trades: List[Dict[str, Any]] = s.get("recent_trades", [])

        if not trades:
            return Panel(
                Align.center("[dim]No closed trades yet[/dim]", vertical="middle"),
                title="[bold white]Trade History[/bold white]",
                border_style="white",
            )

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold dim",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Symbol",      width=12, style="cyan bold")
        table.add_column("Dir",         width=6)
        table.add_column("Entry",       width=12, justify="right")
        table.add_column("Exit",        width=12, justify="right")
        table.add_column("PnL $",       width=12, justify="right")
        table.add_column("PnL %",       width=8,  justify="right")
        table.add_column("Exit Reason", width=14, style="dim")
        table.add_column("Duration",    width=10, justify="right", style="dim")
        table.add_column("Closed",      width=8,  style="dim")

        for trade in reversed(trades[-6:]):
            sym     = trade.get("symbol", "?")
            dirn    = trade.get("direction", "?")
            ep      = float(trade.get("entry_price", 0))
            xp      = float(trade.get("exit_price", 0))
            pnl     = float(trade.get("pnl", 0))
            pnl_pct = float(trade.get("pnl_pct", 0))
            reason  = trade.get("exit_reason", "")
            opened  = trade.get("opened_at")
            closed  = trade.get("closed_at")

            duration_str = "—"
            if opened and closed:
                try:
                    if hasattr(opened, "timestamp") and hasattr(closed, "timestamp"):
                        dur = closed.timestamp() - opened.timestamp()
                        duration_str = _fmt_duration(dur)
                except Exception:
                    pass

            closed_str = "—"
            if closed:
                try:
                    closed_str = closed.strftime("%H:%M") if hasattr(closed, "strftime") else str(closed)[11:16]
                except Exception:
                    pass

            def _fp(v: float) -> str:
                return f"${v:,.0f}" if v >= 1000 else f"${v:,.2f}"

            table.add_row(
                sym,
                Text(dirn.upper(), style=_dir_style(dirn)),
                _fp(ep),
                _fp(xp),
                Text(f"${pnl:+,.2f}", style=_pnl_style(pnl)),
                Text(f"{pnl_pct:+.2%}", style=_pnl_style(pnl_pct)),
                reason,
                duration_str,
                closed_str,
            )

        return Panel(table, title="[bold white]Trade History[/bold white]", border_style="white", padding=(0, 1))

    def _footer(self, s: Dict[str, Any]) -> Panel:
        cycle    = s.get("loop_count", 0)
        errors   = s.get("errors", 0)
        next_in  = s.get("next_cycle_in", 0)
        last_at  = s.get("last_loop_at", "—")
        uptime   = _fmt_duration(time.monotonic() - self._start_time)

        err_style = "bold red" if errors else "dim"
        next_str  = _fmt_duration(next_in) if next_in > 0 else "running…"

        parts = [
            f"[dim]Cycle[/dim] [bold]#{cycle}[/bold]",
            f"[dim]Last run[/dim] [bold]{last_at}[/bold]",
            f"[dim]Next in[/dim] [bold cyan]{next_str}[/bold cyan]",
            f"[dim]Uptime[/dim] [bold]{uptime}[/bold]",
            f"[{err_style}]Errors: {errors}[/{err_style}]",
            "[dim]  Ctrl+C to stop  [/dim]",
        ]
        return Panel(Align.center("  ·  ".join(parts)), style="on grey11", border_style="bright_blue")

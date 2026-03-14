"""
Rich terminal dashboard for live monitoring of the trading agent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


class Dashboard:
    """
    Rich terminal dashboard.

    Usage::

        dashboard = Dashboard()
        with dashboard.live():
            dashboard.update(state)
    """

    def __init__(self, refresh_seconds: float = 10.0) -> None:
        self.refresh_seconds = refresh_seconds
        self.console = Console()
        self._live: Optional[Live] = None
        self._state: Dict[str, Any] = {}

    def live(self) -> Live:
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=1 / self.refresh_seconds,
            screen=False,
        )
        return self._live

    def update(self, state: Dict[str, Any]) -> None:
        self._state = state
        if self._live:
            self._live.update(self._render())

    def print_banner(self) -> None:
        self.console.print(
            Panel.fit(
                "[bold cyan]Auto-Trader-Agent[/bold cyan] · Local LLM Trading System\n"
                "[dim]Powered by Ollama · Multi-Agent Adversarial Decision Framework[/dim]",
                border_style="cyan",
            )
        )

    def print_decision(self, proposal: Any, risk: Any) -> None:
        if not risk.approved:
            self.console.print(
                f"[red]VETOED[/red] {proposal.symbol} — {risk.veto_reason}"
            )
            return
        colour = "green" if proposal.direction == "long" else "red"
        self.console.print(
            f"[{colour}]▶ {proposal.direction.upper()}[/{colour}] "
            f"[bold]{proposal.symbol}[/bold] "
            f"confidence=[yellow]{proposal.confidence:.0%}[/yellow] "
            f"entry={proposal.entry_price:.4f} "
            f"SL={proposal.stop_loss:.4f} "
            f"TP={proposal.take_profit:.4f} "
            f"R:R={proposal.risk_reward:.1f}"
        )

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> Panel:
        s = self._state
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["header"].update(self._header())
        layout["body"].split_row(
            Layout(self._portfolio_panel(s), name="portfolio"),
            Layout(self._positions_panel(s), name="positions"),
            Layout(self._signals_panel(s), name="signals"),
        )
        layout["footer"].update(self._footer(s))
        return Panel(layout, title="[bold cyan]Auto-Trader-Agent[/bold cyan]", border_style="cyan")

    def _header(self) -> Panel:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        return Panel(
            f"[dim]{now}[/dim]  [bold]Mode:[/bold] {self._state.get('mode', 'paper').upper()}  "
            f"[bold]LLM:[/bold] {self._state.get('llm_model', 'unknown')}",
            style="on dark_blue",
        )

    def _portfolio_panel(self, s: Dict[str, Any]) -> Panel:
        perf = s.get("performance", {})
        value = perf.get("portfolio_value", 0)
        ret = perf.get("total_return_pct", 0)
        sharpe = perf.get("sharpe_ratio", 0)
        dd = perf.get("max_drawdown", 0)
        wr = perf.get("win_rate", 0)
        trades = perf.get("total_trades", 0)

        ret_colour = "green" if ret >= 0 else "red"
        lines = [
            f"Value:    [bold]${value:,.2f}[/bold]",
            f"Return:   [{ret_colour}]{ret:+.2%}[/{ret_colour}]",
            f"Sharpe:   {sharpe:.2f}",
            f"Drawdown: [red]{dd:.2%}[/red]",
            f"Win Rate: {wr:.1%}  ({trades} trades)",
        ]
        return Panel("\n".join(lines), title="Portfolio", border_style="blue")

    def _positions_panel(self, s: Dict[str, Any]) -> Panel:
        positions = s.get("positions", {})
        if not positions:
            return Panel("[dim]No open positions[/dim]", title="Positions", border_style="yellow")
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Symbol", style="cyan")
        table.add_column("Dir")
        table.add_column("Entry")
        table.add_column("SL")
        table.add_column("TP")
        for pos_id, pos in list(positions.items())[:5]:
            colour = "green" if pos.get("direction") == "long" else "red"
            table.add_row(
                pos.get("symbol", "?"),
                f"[{colour}]{pos.get('direction', '?').upper()}[/{colour}]",
                f"{pos.get('entry_price', 0):.4f}",
                f"{pos.get('stop_loss', 0):.4f}",
                f"{pos.get('take_profit', 0):.4f}",
            )
        return Panel(table, title="Positions", border_style="yellow")

    def _signals_panel(self, s: Dict[str, Any]) -> Panel:
        signals = s.get("last_signals", [])
        if not signals:
            return Panel("[dim]Waiting for signals...[/dim]", title="Last Signals", border_style="magenta")
        lines = []
        for sig in signals[-6:]:
            colour = "green" if sig.get("direction") == "long" else "red" if sig.get("direction") == "short" else "dim"
            lines.append(
                f"[{colour}]{sig.get('symbol', '?'):12}[/{colour}] "
                f"{sig.get('direction', 'none'):5} "
                f"conf={sig.get('confidence', 0):.0%}"
            )
        return Panel("\n".join(lines), title="Last Signals", border_style="magenta")

    def _footer(self, s: Dict[str, Any]) -> Panel:
        loop = s.get("loop_count", 0)
        last = s.get("last_loop_at", "never")
        errors = s.get("errors", 0)
        return Panel(
            f"Loops: {loop}  Last: {last}  Errors: {errors}  "
            "[dim]Ctrl+C to stop[/dim]",
            style="dim",
        )

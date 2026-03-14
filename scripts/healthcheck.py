"""
Pre-flight health check — run before starting the trader.
Verifies: Ollama, exchange connectivity, config validity.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print("\n[bold cyan]Auto-Trader-Agent Health Check[/bold cyan]\n")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    all_ok = True

    # ── Config ─────────────────────────────────────────────────────────────
    try:
        from config.settings import Settings
        cfg_path = Path("config/config.yaml")
        settings = Settings.from_yaml(cfg_path)
        table.add_row("Config", "[green]OK[/green]", f"mode={settings.trading.mode}")
    except Exception as e:
        table.add_row("Config", "[red]FAIL[/red]", str(e))
        all_ok = False
        settings = None

    # ── Ollama ─────────────────────────────────────────────────────────────
    if settings:
        try:
            from llm.ollama_client import OllamaClient
            client = OllamaClient(base_url=settings.llm.base_url, model=settings.llm.model)
            healthy = await client.health_check()
            models = await client.list_models()
            if healthy:
                model_names = ", ".join(m.split(":")[0] for m in models[:5])
                table.add_row("Ollama", "[green]OK[/green]", f"models: {model_names or 'none pulled'}")
            else:
                table.add_row("Ollama", "[yellow]OFFLINE[/yellow]", f"run: ollama serve")
                all_ok = False
        except Exception as e:
            table.add_row("Ollama", "[red]FAIL[/red]", str(e))
            all_ok = False

    # ── Exchange ────────────────────────────────────────────────────────────
    if settings:
        try:
            import ccxt.async_support as ccxt
            exchange_class = getattr(ccxt, settings.exchange.id)
            exchange = exchange_class({"sandbox": settings.exchange.sandbox})
            markets = await exchange.load_markets()
            await exchange.close()
            n = len(markets)
            table.add_row(
                "Exchange",
                "[green]OK[/green]",
                f"{settings.exchange.id} (sandbox={settings.exchange.sandbox}) — {n} markets",
            )
        except Exception as e:
            table.add_row("Exchange", "[yellow]WARN[/yellow]", f"offline or rate-limited: {e}")

    # ── Dependencies ───────────────────────────────────────────────────────
    deps = ["ccxt", "pandas", "numpy", "ta", "httpx", "pydantic", "rich", "loguru", "aiosqlite"]
    missing = []
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing.append(dep)
    if missing:
        table.add_row("Dependencies", "[red]MISSING[/red]", ", ".join(missing))
        all_ok = False
    else:
        table.add_row("Dependencies", "[green]OK[/green]", f"{len(deps)} packages present")

    console.print(table)

    if all_ok:
        console.print("\n[bold green]All checks passed — ready to trade![/bold green]\n")
    else:
        console.print("\n[bold yellow]Some checks failed — see table above.[/bold yellow]\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

"""Rich table renderers for predictions and reports."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from betbot.display.formatters import (
    fmt_confidence,
    fmt_edge,
    fmt_market,
    fmt_odds,
    fmt_prob,
    fmt_selection,
)
from betbot.models.ensemble import MatchPrediction

console = Console()


def render_match_predictions(
    prediction: MatchPrediction,
    home_name: str,
    away_name: str,
    match_date: str,
    league_name: str,
    market_filter: str | None = None,
    min_edge: float = 0.0,
) -> None:
    """Print a formatted match prediction card to the console."""

    results = prediction.results
    if market_filter:
        results = [r for r in results if r.market == market_filter]
    if min_edge > 0:
        results = [r for r in results if r.edge is not None and r.edge >= min_edge]

    if not results:
        console.print(f"[dim]No predictions meet the filter criteria.[/]")
        return

    title = f"[bold]{home_name}[/] vs [bold]{away_name}[/]"
    subtitle = f"{league_name} · {match_date[:16].replace('T', '  ')}"

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    table.add_column("Market", style="bold", min_width=12)
    table.add_column("Selection", min_width=14)
    table.add_column("Model Prob", justify="right", min_width=10)
    table.add_column("Bk Odds", justify="right", min_width=8)
    table.add_column("Edge", justify="right", min_width=8)
    table.add_column("Confidence", justify="center", min_width=12)

    # Group by market for cleaner display
    seen_markets: set[str] = set()
    for r in sorted(results, key=lambda x: (x.market, x.selection)):
        market_label = fmt_market(r.market) if r.market not in seen_markets else ""
        seen_markets.add(r.market)

        table.add_row(
            market_label,
            fmt_selection(r.selection),
            fmt_prob(r.model_prob),
            fmt_odds(r.bookmaker_odds),
            fmt_edge(r.edge),
            fmt_confidence(r.confidence),
        )

    console.print(Panel(table, title=title, subtitle=subtitle, border_style="blue"))


def render_sync_report(
    fixtures_synced: int,
    stats_synced: int,
    odds_synced: int,
    calls_used: int,
    calls_remaining: int,
    errors: list[str],
) -> None:
    table = Table(title="Sync Report", box=box.SIMPLE_HEAD, show_header=False)
    table.add_column("Item", style="bold")
    table.add_column("Value")

    table.add_row("Fixtures synced", str(fixtures_synced))
    table.add_row("Stats synced", str(stats_synced))
    table.add_row("Odds synced", str(odds_synced))
    table.add_row("API calls used today", str(calls_used))
    table.add_row("API calls remaining", f"[{'green' if calls_remaining > 20 else 'red'}]{calls_remaining}[/]")

    console.print(table)

    for err in errors:
        console.print(f"[yellow]⚠ {err}[/]")


def render_budget(used: int, remaining: int, limit: int) -> None:
    pct = (used / limit) * 100
    bar_width = 40
    filled = int(bar_width * used / limit)
    bar = "█" * filled + "░" * (bar_width - filled)
    color = "green" if remaining > 30 else ("yellow" if remaining > 10 else "red")

    console.print(f"\n[bold]API Budget Today[/]")
    console.print(f"[{color}]{bar}[/] {used}/{limit} calls used ({pct:.0f}%)")
    console.print(f"Remaining: [{color}]{remaining}[/] calls\n")


def render_training_summary(model_name: str, metrics: dict) -> None:
    console.print(f"\n[bold green]✓[/] Trained [bold]{model_name}[/]")
    for k, v in metrics.items():
        console.print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

"""Rich formatting helpers."""

from __future__ import annotations

from rich.style import Style
from rich.text import Text


CONFIDENCE_BADGES = {
    "very_high": "[bold green]★★★★[/]",
    "high":      "[bold green]★★★[/]",
    "medium":    "[bold yellow]★★[/]",
    "low":       "[yellow]★[/]",
    "none":      "[dim]—[/]",
}

MARKET_LABELS = {
    "goals_ou":  "Goals O/U",
    "1x2":       "1X2",
    "btts":      "BTTS",
    "cards_ou":  "Cards O/U",
    "corners_ou":"Corners O/U",
}

SELECTION_LABELS = {
    "over_1.5": "Over 1.5",
    "under_1.5": "Under 1.5",
    "over_2.5": "Over 2.5",
    "under_2.5": "Under 2.5",
    "over_3.5": "Over 3.5",
    "under_3.5": "Under 3.5",
    "over_4.5": "Over 4.5",
    "over_8.5": "Over 8.5",
    "over_9.5": "Over 9.5",
    "over_10.5": "Over 10.5",
    "over_11.5": "Over 11.5",
    "home": "Home Win",
    "draw": "Draw",
    "away": "Away Win",
    "yes": "Yes",
    "no": "No",
    "any_red": "Any Red Card",
}


def fmt_prob(p: float) -> str:
    return f"{p * 100:.1f}%"


def fmt_odds(o: float | None) -> str:
    if o is None:
        return "—"
    return f"{o:.2f}"


def fmt_edge(edge: float | None) -> Text:
    if edge is None:
        return Text("—", style="dim")
    pct = edge * 100
    if pct >= 8:
        return Text(f"+{pct:.1f}%", style="bold green")
    if pct >= 3:
        return Text(f"+{pct:.1f}%", style="green")
    if pct > 0:
        return Text(f"+{pct:.1f}%", style="yellow")
    return Text(f"{pct:.1f}%", style="red")


def fmt_confidence(level: str) -> str:
    return CONFIDENCE_BADGES.get(level, "—")


def fmt_market(market: str) -> str:
    return MARKET_LABELS.get(market, market)


def fmt_selection(selection: str) -> str:
    return SELECTION_LABELS.get(selection, selection.replace("_", " ").title())

"""Edge calculation and confidence scoring."""

from __future__ import annotations

from betbot.config import (
    EDGE_HIGH,
    EDGE_LOW,
    EDGE_MEDIUM,
    EDGE_VERY_HIGH,
    KELLY_CAP,
)
from betbot.utils.math_helpers import implied_probability, kelly_fraction


def calculate_edge(model_prob: float, bookmaker_odds: float) -> float:
    """
    Edge = model probability - bookmaker implied probability.
    Positive means the model sees value over the bookmaker price.
    """
    return model_prob - implied_probability(bookmaker_odds)


def assign_confidence(edge: float) -> str:
    if edge >= EDGE_VERY_HIGH:
        return "very_high"
    if edge >= EDGE_HIGH:
        return "high"
    if edge >= EDGE_MEDIUM:
        return "medium"
    if edge >= EDGE_LOW:
        return "low"
    return "none"


def kelly_stake(model_prob: float, bookmaker_odds: float, bankroll: float = 1000.0) -> float:
    """Suggested Kelly stake in currency units."""
    fraction = kelly_fraction(model_prob, bookmaker_odds, cap=KELLY_CAP)
    return round(fraction * bankroll, 2)

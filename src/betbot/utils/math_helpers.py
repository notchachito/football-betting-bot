"""Mathematical utilities for probability models."""

from __future__ import annotations

import numpy as np
from scipy import stats


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for a Poisson distribution."""
    return float(stats.poisson.pmf(k, lam))


def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for a Poisson distribution."""
    return float(stats.poisson.cdf(k, lam))


def dixon_coles_tau(x: int, y: int, lam1: float, lam2: float, rho: float) -> float:
    """
    Dixon-Coles correction factor for low-scoring outcomes.

    Corrects the independence assumption of the basic Poisson model for
    the four low-score cells: (0,0), (1,0), (0,1), (1,1).
    """
    if x == 0 and y == 0:
        return 1 - lam1 * lam2 * rho
    if x == 1 and y == 0:
        return 1 + lam2 * rho
    if x == 0 and y == 1:
        return 1 + lam1 * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def build_score_matrix(lam1: float, lam2: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """
    Build an (max_goals+1) x (max_goals+1) probability matrix where
    matrix[i, j] = P(home scores i, away scores j).
    """
    size = max_goals + 1
    matrix = np.zeros((size, size), dtype=np.float64)

    for i in range(size):
        for j in range(size):
            tau = dixon_coles_tau(i, j, lam1, lam2, rho)
            matrix[i, j] = tau * poisson_pmf(i, lam1) * poisson_pmf(j, lam2)

    # Normalise so probabilities sum to 1
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def remove_vig(probs: list[float]) -> list[float]:
    """Normalise a list of bookmaker probabilities to remove the overround."""
    total = sum(probs)
    if total <= 0:
        return probs
    return [p / total for p in probs]


def kelly_fraction(model_prob: float, decimal_odds: float, cap: float = 0.05) -> float:
    """
    Full Kelly criterion fraction.
    Returns 0 if there is no edge (negative Kelly).
    Capped at `cap` to avoid over-betting.
    """
    if decimal_odds <= 1.0 or model_prob <= 0:
        return 0.0
    b = decimal_odds - 1  # net fractional odds
    f = (model_prob * (b + 1) - 1) / b
    return min(max(f, 0.0), cap)


def time_decay_weight(days_ago: float, xi: float = 0.005) -> float:
    """Exponential time-decay weight. More recent = higher weight."""
    return float(np.exp(-xi * days_ago))

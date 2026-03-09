"""Tests for math_helpers module."""

import numpy as np
import pytest
from scipy import stats

from betbot.utils.math_helpers import (
    build_score_matrix,
    dixon_coles_tau,
    implied_probability,
    kelly_fraction,
    poisson_cdf,
    poisson_pmf,
    remove_vig,
    time_decay_weight,
)


def test_poisson_pmf_matches_scipy():
    for k in range(6):
        for lam in [0.5, 1.0, 1.5, 2.5]:
            assert abs(poisson_pmf(k, lam) - stats.poisson.pmf(k, lam)) < 1e-10


def test_poisson_cdf_matches_scipy():
    for k in range(6):
        for lam in [0.5, 1.0, 2.5]:
            assert abs(poisson_cdf(k, lam) - stats.poisson.cdf(k, lam)) < 1e-10


def test_dixon_coles_tau_non_low_scores():
    assert dixon_coles_tau(2, 3, 1.5, 1.2, -0.03) == 1.0
    assert dixon_coles_tau(5, 0, 2.0, 1.0, -0.03) == 1.0


def test_dixon_coles_tau_low_scores():
    lam1, lam2, rho = 1.5, 1.2, -0.03
    assert abs(dixon_coles_tau(0, 0, lam1, lam2, rho) - (1 - lam1 * lam2 * rho)) < 1e-10
    assert abs(dixon_coles_tau(1, 0, lam1, lam2, rho) - (1 + lam2 * rho)) < 1e-10
    assert abs(dixon_coles_tau(0, 1, lam1, lam2, rho) - (1 + lam1 * rho)) < 1e-10
    assert abs(dixon_coles_tau(1, 1, lam1, lam2, rho) - (1 - rho)) < 1e-10


def test_score_matrix_sums_to_one():
    matrix = build_score_matrix(1.5, 1.2, -0.03)
    assert abs(matrix.sum() - 1.0) < 1e-6


def test_score_matrix_shape():
    matrix = build_score_matrix(1.5, 1.2, -0.03)
    assert matrix.shape == (11, 11)


def test_score_matrix_non_negative():
    matrix = build_score_matrix(1.5, 1.2, -0.03)
    assert np.all(matrix >= 0)


def test_implied_probability():
    assert abs(implied_probability(2.0) - 0.5) < 1e-10
    assert abs(implied_probability(1.5) - (1 / 1.5)) < 1e-10
    assert implied_probability(1.0) == 1.0


def test_remove_vig_normalises():
    probs = [0.45, 0.30, 0.35]  # sum = 1.10 (10% vig)
    fair = remove_vig(probs)
    assert abs(sum(fair) - 1.0) < 1e-10
    # Ratios preserved
    assert abs(fair[0] / fair[1] - probs[0] / probs[1]) < 1e-6


def test_kelly_fraction_positive_edge():
    # Model says 60% chance, odds are 2.0 (implied 50%)
    f = kelly_fraction(0.60, 2.0, cap=1.0)
    assert f > 0
    # Full Kelly: (0.6 * 2 - 1) / (2 - 1) = 0.2
    assert abs(f - 0.2) < 1e-10


def test_kelly_fraction_no_edge():
    # Fair odds
    f = kelly_fraction(0.50, 2.0)
    assert f == 0.0


def test_kelly_fraction_capped():
    f = kelly_fraction(0.90, 2.0, cap=0.05)
    assert f == 0.05


def test_time_decay_recent():
    w0 = time_decay_weight(0)
    w30 = time_decay_weight(30)
    w180 = time_decay_weight(180)
    assert w0 == pytest.approx(1.0)
    assert w0 > w30 > w180
    assert w180 > 0

"""Tests for value.py — edge and Kelly calculations."""

from betbot.markets.value import assign_confidence, calculate_edge, kelly_stake


def test_positive_edge():
    edge = calculate_edge(0.60, 2.0)  # model 60%, implied 50%
    assert abs(edge - 0.10) < 1e-10


def test_negative_edge():
    edge = calculate_edge(0.45, 2.0)  # model 45%, implied 50%
    assert edge < 0


def test_zero_edge():
    edge = calculate_edge(0.50, 2.0)
    assert abs(edge) < 1e-10


def test_confidence_very_high():
    assert assign_confidence(0.13) == "very_high"


def test_confidence_high():
    assert assign_confidence(0.09) == "high"


def test_confidence_medium():
    assert assign_confidence(0.06) == "medium"


def test_confidence_low():
    assert assign_confidence(0.04) == "low"


def test_confidence_none():
    assert assign_confidence(0.01) == "none"
    assert assign_confidence(-0.05) == "none"


def test_kelly_stake_positive():
    stake = kelly_stake(0.60, 2.0, bankroll=1000)
    assert stake > 0
    assert stake <= 50  # capped at 5%


def test_kelly_stake_no_edge():
    stake = kelly_stake(0.50, 2.0, bankroll=1000)
    assert stake == 0.0

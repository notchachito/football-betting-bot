"""Tests for the Dixon-Coles model."""

import numpy as np
import pytest

from betbot.models.dixon_coles import DixonColesModel
from betbot.models.features import build_dixon_coles_dataset


def test_fit_requires_minimum_matches(sample_matches):
    df = build_dixon_coles_dataset(sample_matches[:5])
    with pytest.raises(ValueError, match="at least 20"):
        DixonColesModel().fit(df)


def test_fit_converges(sample_matches):
    df = build_dixon_coles_dataset(sample_matches)
    model = DixonColesModel().fit(df)
    assert model.is_fitted


def test_score_matrix_sums_to_one(sample_matches):
    df = build_dixon_coles_dataset(sample_matches)
    model = DixonColesModel().fit(df)
    team_ids = model.team_ids()
    matrix = model.predict_score_matrix(team_ids[0], team_ids[1])
    assert abs(matrix.sum() - 1.0) < 1e-4


def test_1x2_probs_sum_to_one(sample_matches):
    df = build_dixon_coles_dataset(sample_matches)
    model = DixonColesModel().fit(df)
    team_ids = model.team_ids()
    r = model.predict_1x2(team_ids[0], team_ids[1])
    total = r.home_win + r.draw + r.away_win
    assert abs(total - 1.0) < 1e-4


def test_btts_probability_in_range(sample_matches):
    df = build_dixon_coles_dataset(sample_matches)
    model = DixonColesModel().fit(df)
    team_ids = model.team_ids()
    btts = model.predict_btts(team_ids[0], team_ids[1])
    assert 0.0 <= btts <= 1.0


def test_goals_over_under_complement(sample_matches):
    df = build_dixon_coles_dataset(sample_matches)
    model = DixonColesModel().fit(df)
    team_ids = model.team_ids()
    goals = model.predict_goals(team_ids[0], team_ids[1])
    assert abs(goals.over_2_5 + goals.under_2_5 - 1.0) < 1e-4
    assert abs(goals.over_1_5 + goals.under_1_5 - 1.0) < 1e-4
    assert abs(goals.over_3_5 + goals.under_3_5 - 1.0) < 1e-4


def test_unknown_team_uses_league_average(sample_matches):
    df = build_dixon_coles_dataset(sample_matches)
    model = DixonColesModel().fit(df)
    # Team 99999 is unknown — should not raise, should use default params
    r = model.predict_1x2(99999, 99998)
    assert 0 < r.home_win < 1
    assert 0 < r.draw < 1
    assert 0 < r.away_win < 1


def test_serialise_deserialise(sample_matches):
    df = build_dixon_coles_dataset(sample_matches)
    model = DixonColesModel().fit(df)
    team_ids = model.team_ids()

    before = model.predict_1x2(team_ids[0], team_ids[1])
    restored = DixonColesModel.from_params(model.get_params())
    after = restored.predict_1x2(team_ids[0], team_ids[1])

    assert abs(before.home_win - after.home_win) < 1e-10
    assert abs(before.draw - after.draw) < 1e-10


def test_home_advantage(sample_matches):
    """Home team should generally have higher win probability."""
    df = build_dixon_coles_dataset(sample_matches)
    model = DixonColesModel().fit(df)
    team_ids = model.team_ids()
    # Over many team combinations, home advantage should hold on average
    home_wins = []
    for i in range(min(6, len(team_ids))):
        for j in range(min(6, len(team_ids))):
            if i != j:
                r = model.predict_1x2(team_ids[i], team_ids[j])
                r_rev = model.predict_1x2(team_ids[j], team_ids[i])
                home_wins.append(r.home_win > r_rev.away_win)
    # At least 60% of matchups should show home advantage
    assert sum(home_wins) / len(home_wins) >= 0.6

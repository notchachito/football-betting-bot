"""Tests for historical data helpers."""

import pandas as pd
import pytest

from betbot.data.historical import _parse_fdfd, _season_str, build_referee_profiles


def test_season_str():
    assert _season_str(2024) == "2425"
    assert _season_str(2023) == "2324"
    assert _season_str(2022) == "2223"


def test_parse_fdfd_maps_columns():
    df = pd.DataFrame({
        "HomeTeam": ["Arsenal", "Chelsea"],
        "AwayTeam": ["Chelsea", "Arsenal"],
        "FTHG": [2, 1],
        "FTAG": [1, 2],
        "HC": [6, 4],
        "AC": [3, 7],
        "HY": [2, 1],
        "AY": [3, 2],
        "HR": [0, 1],
        "AR": [0, 0],
        "HF": [12, 10],
        "AF": [11, 13],
        "Referee": ["M. Oliver", "M. Dean"],
        "Date": ["2024-01-01", "2024-01-08"],
        "B365H": [1.90, 2.10],
        "B365D": [3.50, 3.40],
        "B365A": [4.20, 3.80],
    })
    result = _parse_fdfd(df, "Premier League")
    assert len(result) == 2
    assert "home_goals" in result.columns
    assert "home_corners" in result.columns
    assert "referee" in result.columns
    assert "odds_home" in result.columns


def test_parse_fdfd_drops_missing_goals():
    df = pd.DataFrame({
        "HomeTeam": ["Arsenal", "Chelsea"],
        "AwayTeam": ["Chelsea", "Arsenal"],
        "FTHG": [2, None],
        "FTAG": [1, 0],
        "HC": [6, 4],
        "AC": [3, 7],
        "HY": [2, 1],
        "AY": [3, 2],
        "HR": [0, 0],
        "AR": [0, 0],
        "HF": [12, 10],
        "AF": [11, 13],
    })
    result = _parse_fdfd(df, "Premier League")
    assert len(result) == 1


def test_build_referee_profiles():
    df = pd.DataFrame({
        "referee": ["M. Oliver"] * 15 + ["M. Dean"] * 12,
        "home_yellows": [2, 3, 1, 2, 2, 3, 1, 2, 3, 2, 1, 3, 2, 2, 1,
                         1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1],
        "away_yellows": [3, 2, 2, 3, 1, 2, 3, 2, 1, 3, 2, 2, 3, 1, 2,
                         2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1],
        "home_fouls": [14] * 27,
        "away_fouls": [12] * 27,
    })
    profiles = build_referee_profiles(df)
    assert "M. Oliver" in profiles.index
    assert "M. Dean" in profiles.index
    assert profiles.loc["M. Oliver", "avg_total_cards"] > 0
    assert "strictness_score" in profiles.columns
    assert "cards_per_foul" in profiles.columns
    assert "n_matches" in profiles.columns


def test_referee_profiles_empty_on_missing_data():
    df = pd.DataFrame({"home_goals": [1, 2], "away_goals": [0, 1]})
    result = build_referee_profiles(df)
    assert result.empty


def test_referee_strictness_score_direction():
    """A stricter referee should have a higher strictness score."""
    df = pd.DataFrame({
        "referee": ["Strict"] * 20 + ["Lenient"] * 20,
        "home_yellows": [4] * 20 + [1] * 20,
        "away_yellows": [3] * 20 + [1] * 20,
        "home_fouls": [14] * 40,
        "away_fouls": [12] * 40,
    })
    profiles = build_referee_profiles(df)
    assert profiles.loc["Strict", "strictness_score"] > profiles.loc["Lenient", "strictness_score"]

"""Tests for XGBoost cards, Poisson corners, and Logistic BTTS models."""

import pytest

from betbot.models.features import (
    build_btts_features,
    build_cards_features,
    build_corners_features,
)
from betbot.models.logistic_btts import LogisticBTTSModel
from betbot.models.poisson_glm import PoissonCornersModel
from betbot.models.xgboost_cards import XGBoostCardsModel


# ---------------------------------------------------------------------------
# XGBoost Cards
# ---------------------------------------------------------------------------

class TestXGBoostCards:
    def test_fit_requires_minimum(self, sample_matches, sample_stats):
        df = build_cards_features(sample_matches[:5], sample_stats[:5])
        with pytest.raises((ValueError, Exception)):
            XGBoostCardsModel().fit(df)

    def test_fit_and_predict(self, sample_matches, sample_stats):
        df = build_cards_features(sample_matches, sample_stats)
        if len(df) < 50:
            pytest.skip("Not enough samples for this fixture set")
        model = XGBoostCardsModel().fit(df)
        pred = model.predict(
            home_yellows_avg=2.0, away_yellows_avg=2.5,
            home_fouls_avg=12.0, away_fouls_avg=13.0,
        )
        assert 0 < pred.expected_yellows < 15
        assert 0.0 <= pred.p_over_3_5 <= 1.0
        assert 0.0 <= pred.p_over_4_5 <= 1.0
        assert 0.0 <= pred.p_any_red <= 1.0
        assert pred.p_over_4_5 <= pred.p_over_3_5

    def test_serialise(self, sample_matches, sample_stats):
        df = build_cards_features(sample_matches, sample_stats)
        if len(df) < 50:
            pytest.skip("Not enough samples")
        model = XGBoostCardsModel().fit(df)
        pred1 = model.predict(2.0, 2.5, 12.0, 13.0)
        restored = XGBoostCardsModel.from_params(model.get_params())
        pred2 = restored.predict(2.0, 2.5, 12.0, 13.0)
        assert abs(pred1.expected_yellows - pred2.expected_yellows) < 0.001

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            XGBoostCardsModel().predict(2.0, 2.0, 12.0, 12.0)


# ---------------------------------------------------------------------------
# Poisson Corners
# ---------------------------------------------------------------------------

class TestPoissonCorners:
    def test_fit_and_predict(self, sample_matches, sample_stats):
        df = build_corners_features(sample_matches, sample_stats)
        if len(df) < 50:
            pytest.skip("Not enough samples")
        model = PoissonCornersModel().fit(df)
        pred = model.predict(
            home_corners_for=5.5,
            home_corners_against=4.5,
            away_corners_for=4.8,
            away_corners_against=5.2,
        )
        assert pred.expected_total > 0
        assert 0.0 <= pred.p_over_9_5 <= 1.0
        assert 0.0 <= pred.p_over_10_5 <= 1.0
        assert pred.p_over_10_5 <= pred.p_over_9_5

    def test_serialise(self, sample_matches, sample_stats):
        df = build_corners_features(sample_matches, sample_stats)
        if len(df) < 50:
            pytest.skip("Not enough samples")
        model = PoissonCornersModel().fit(df)
        pred1 = model.predict(5.0, 4.5, 5.0, 4.5)
        restored = PoissonCornersModel.from_params(model.get_params())
        pred2 = restored.predict(5.0, 4.5, 5.0, 4.5)
        assert abs(pred1.expected_total - pred2.expected_total) < 0.001

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            PoissonCornersModel().predict(5.0, 4.5, 5.0, 4.5)


# ---------------------------------------------------------------------------
# Logistic BTTS
# ---------------------------------------------------------------------------

class TestLogisticBTTS:
    def test_fit_and_predict(self, sample_matches):
        df = build_btts_features(sample_matches)
        if len(df) < 50:
            pytest.skip("Not enough samples")
        model = LogisticBTTSModel().fit(df)
        prob = model.predict_proba(0.7, 0.6, 0.65, 0.55)
        assert 0.0 <= prob <= 1.0

    def test_high_scoring_teams_higher_btts(self, sample_matches):
        df = build_btts_features(sample_matches)
        if len(df) < 50:
            pytest.skip("Not enough samples")
        model = LogisticBTTSModel().fit(df)
        # Teams that always score and always concede should have higher BTTS
        p_high = model.predict_proba(1.0, 1.0, 1.0, 1.0)
        p_low = model.predict_proba(0.0, 0.0, 0.0, 0.0)
        assert p_high > p_low

    def test_serialise(self, sample_matches):
        df = build_btts_features(sample_matches)
        if len(df) < 50:
            pytest.skip("Not enough samples")
        model = LogisticBTTSModel().fit(df)
        p1 = model.predict_proba(0.7, 0.6, 0.65, 0.55)
        restored = LogisticBTTSModel.from_params(model.get_params())
        p2 = restored.predict_proba(0.7, 0.6, 0.65, 0.55)
        assert abs(p1 - p2) < 0.001

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            LogisticBTTSModel().predict_proba(0.7, 0.6, 0.65, 0.55)

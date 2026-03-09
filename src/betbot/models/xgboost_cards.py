"""XGBoost model for predicting yellow cards and red card probability."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from betbot.utils.math_helpers import poisson_cdf

FEATURE_COLS = [
    "home_yellows_avg",
    "away_yellows_avg",
    "home_fouls_avg",
    "away_fouls_avg",
    "combined_yellows_avg",
    "combined_fouls_avg",
    # Referee features — #1 importance group (research confirmed)
    "ref_avg_cards",
    "ref_strictness",
    "ref_cards_per_foul",
]


@dataclass(frozen=True)
class CardsPrediction:
    expected_yellows: float
    p_over_3_5: float
    p_over_4_5: float
    p_any_red: float


class XGBoostCardsModel:
    MODEL_NAME = "xgboost_cards"

    def __init__(self) -> None:
        self._yellows_model: XGBRegressor | None = None
        self._red_model: CalibratedClassifierCV | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, features_df: pd.DataFrame) -> "XGBoostCardsModel":
        if len(features_df) < 50:
            raise ValueError(f"Need at least 50 matches with stats; got {len(features_df)}")

        df = features_df.sort_values("match_date").reset_index(drop=True)
        X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_yellows = df["total_yellows"].to_numpy(dtype=np.float32)
        y_red = (df["any_red"] > 0).astype(int).to_numpy()

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_yt, y_yv = y_yellows[:split], y_yellows[split:]
        y_rt, y_rv = y_red[:split], y_red[split:]

        self._yellows_model = XGBRegressor(
            max_depth=4,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self._yellows_model.fit(
            X_train, y_yt,
            eval_set=[(X_val, y_yv)],
            verbose=False,
        )

        base_red = XGBClassifier(
            max_depth=3,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        tscv = TimeSeriesSplit(n_splits=5)
        self._red_model = CalibratedClassifierCV(base_red, cv=tscv, method="isotonic")
        self._red_model.fit(X, y_red)

        self._is_fitted = True
        return self

    def predict(
        self,
        home_yellows_avg: float,
        away_yellows_avg: float,
        home_fouls_avg: float,
        away_fouls_avg: float,
        ref_avg_cards: float = 3.5,
        ref_strictness: float = 0.0,
        ref_cards_per_foul: float = 0.12,
    ) -> CardsPrediction:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        x = np.array([[
            home_yellows_avg, away_yellows_avg,
            home_fouls_avg, away_fouls_avg,
            home_yellows_avg + away_yellows_avg,
            home_fouls_avg + away_fouls_avg,
            ref_avg_cards, ref_strictness, ref_cards_per_foul,
        ]], dtype=np.float32)

        expected = float(self._yellows_model.predict(x)[0])  # type: ignore[union-attr]
        expected = max(0.5, expected)

        p_over_3_5 = 1.0 - poisson_cdf(3, expected)
        p_over_4_5 = 1.0 - poisson_cdf(4, expected)
        red_proba = self._red_model.predict_proba(x)  # type: ignore[union-attr]
        # If only one class in training data, proba has shape (n, 1)
        p_any_red = float(red_proba[0, 1]) if red_proba.shape[1] > 1 else 0.05

        return CardsPrediction(
            expected_yellows=expected,
            p_over_3_5=p_over_3_5,
            p_over_4_5=p_over_4_5,
            p_any_red=p_any_red,
        )

    def get_params(self) -> bytes:
        return pickle.dumps({
            "yellows": self._yellows_model,
            "red": self._red_model,
        })

    @classmethod
    def from_params(cls, data: bytes) -> "XGBoostCardsModel":
        m = cls()
        p = pickle.loads(data)
        m._yellows_model = p["yellows"]
        m._red_model = p["red"]
        m._is_fitted = True
        return m

"""Poisson GLM for corners prediction."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from betbot.utils.math_helpers import poisson_cdf

FEATURE_COLS = [
    "home_corners_for",
    "home_corners_against",
    "away_corners_for",
    "away_corners_against",
    "combined_corners_for",
]


@dataclass(frozen=True)
class CornersPrediction:
    expected_total: float
    p_over_8_5: float
    p_over_9_5: float
    p_over_10_5: float
    p_over_11_5: float


class PoissonCornersModel:
    MODEL_NAME = "poisson_corners"

    def __init__(self) -> None:
        self._model: PoissonRegressor | None = None
        self._scaler: StandardScaler | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, features_df: pd.DataFrame) -> "PoissonCornersModel":
        if len(features_df) < 50:
            raise ValueError(f"Need at least 50 matches with corner stats; got {len(features_df)}")

        df = features_df.sort_values("match_date").reset_index(drop=True)
        X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
        y = df["total_corners"].to_numpy(dtype=np.float64)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = PoissonRegressor(alpha=1.0, max_iter=500)
        self._model.fit(X_scaled, y)

        self._is_fitted = True
        return self

    def predict(
        self,
        home_corners_for: float,
        home_corners_against: float,
        away_corners_for: float,
        away_corners_against: float,
    ) -> CornersPrediction:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        x = np.array([[
            home_corners_for,
            home_corners_against,
            away_corners_for,
            away_corners_against,
            home_corners_for + away_corners_for,
        ]])

        x_scaled = self._scaler.transform(x)  # type: ignore[union-attr]
        expected = float(self._model.predict(x_scaled)[0])  # type: ignore[union-attr]
        expected = max(4.0, expected)

        return CornersPrediction(
            expected_total=expected,
            p_over_8_5=1.0 - poisson_cdf(8, expected),
            p_over_9_5=1.0 - poisson_cdf(9, expected),
            p_over_10_5=1.0 - poisson_cdf(10, expected),
            p_over_11_5=1.0 - poisson_cdf(11, expected),
        )

    def get_params(self) -> bytes:
        return pickle.dumps({"model": self._model, "scaler": self._scaler})

    @classmethod
    def from_params(cls, data: bytes) -> "PoissonCornersModel":
        m = cls()
        p = pickle.loads(data)
        m._model = p["model"]
        m._scaler = p["scaler"]
        m._is_fitted = True
        return m

"""Logistic regression model for Both Teams To Score prediction."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "home_scored_rate",
    "home_conceded_rate",
    "away_scored_rate",
    "away_conceded_rate",
]


class LogisticBTTSModel:
    MODEL_NAME = "logistic_btts"

    def __init__(self) -> None:
        self._model: CalibratedClassifierCV | None = None
        self._scaler: StandardScaler | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, features_df: pd.DataFrame) -> "LogisticBTTSModel":
        if len(features_df) < 50:
            raise ValueError(f"Need at least 50 matches; got {len(features_df)}")

        df = features_df.sort_values("match_date").reset_index(drop=True)
        X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
        y = df["btts"].to_numpy(dtype=int)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        base = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self._model = CalibratedClassifierCV(base, cv=5, method="isotonic")
        self._model.fit(X_scaled, y)

        self._is_fitted = True
        return self

    def predict_proba(
        self,
        home_scored_rate: float,
        home_conceded_rate: float,
        away_scored_rate: float,
        away_conceded_rate: float,
    ) -> float:
        """Return P(BTTS = Yes)."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        x = np.array([[
            home_scored_rate, home_conceded_rate,
            away_scored_rate, away_conceded_rate,
        ]])
        x_scaled = self._scaler.transform(x)  # type: ignore[union-attr]
        return float(self._model.predict_proba(x_scaled)[0, 1])  # type: ignore[union-attr]

    def get_params(self) -> bytes:
        return pickle.dumps({"model": self._model, "scaler": self._scaler})

    @classmethod
    def from_params(cls, data: bytes) -> "LogisticBTTSModel":
        m = cls()
        p = pickle.loads(data)
        m._model = p["model"]
        m._scaler = p["scaler"]
        m._is_fitted = True
        return m

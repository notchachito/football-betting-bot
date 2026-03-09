"""Dixon-Coles bivariate Poisson model for goals and match result prediction."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from betbot.utils.math_helpers import (
    build_score_matrix,
    poisson_cdf,
)


@dataclass(frozen=True)
class GoalsProbabilities:
    over_1_5: float
    under_1_5: float
    over_2_5: float
    under_2_5: float
    over_3_5: float
    under_3_5: float
    expected_home: float
    expected_away: float


@dataclass(frozen=True)
class MatchResultProbabilities:
    home_win: float
    draw: float
    away_win: float


class DixonColesModel:
    """
    Dixon-Coles (1997) bivariate Poisson model with low-score correction.

    Parameters fitted via Maximum Likelihood Estimation with time-decay weights.
    """

    MODEL_NAME = "dixon_coles"

    def __init__(self) -> None:
        self._attack: dict[int, float] = {}
        self._defense: dict[int, float] = {}
        self._home_adv: float = 1.2
        self._rho: float = -0.03
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, matches_df: pd.DataFrame) -> "DixonColesModel":
        """
        Fit Dixon-Coles parameters from historical match data.

        matches_df must have columns:
            home_team, away_team, home_goals, away_goals, weight
        """
        if len(matches_df) < 20:
            raise ValueError(f"Need at least 20 matches to fit; got {len(matches_df)}")

        teams = sorted(set(matches_df["home_team"]) | set(matches_df["away_team"]))
        n = len(teams)
        team_index = {t: i for i, t in enumerate(teams)}

        # Initial parameter vector: [alpha_0..alpha_n, beta_0..beta_n, home_adv, rho]
        x0 = np.ones(2 * n + 2, dtype=np.float64)
        x0[2 * n] = 1.2   # home advantage
        x0[2 * n + 1] = -0.03  # rho

        bounds = (
            [(0.1, 5.0)] * n      # attack strengths
            + [(0.1, 5.0)] * n    # defense strengths
            + [(1.0, 2.0)]        # home advantage
            + [(-0.15, 0.15)]     # rho
        )

        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(matches_df, team_index, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            # Fallback: relax tolerance
            result = minimize(
                _neg_log_likelihood,
                x0,
                args=(matches_df, team_index, n),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-6},
            )

        params = result.x
        self._attack = {t: params[team_index[t]] for t in teams}
        self._defense = {t: params[n + team_index[t]] for t in teams}
        self._home_adv = float(params[2 * n])
        self._rho = float(params[2 * n + 1])
        self._is_fitted = True

        return self

    def predict_score_matrix(self, home_id: int, away_id: int) -> np.ndarray:
        """11x11 matrix: matrix[i,j] = P(home scores i, away scores j)."""
        self._check_fitted()
        lam1, lam2 = self._lambdas(home_id, away_id)
        return build_score_matrix(lam1, lam2, self._rho)

    def predict_goals(self, home_id: int, away_id: int) -> GoalsProbabilities:
        matrix = self.predict_score_matrix(home_id, away_id)
        lam1, lam2 = self._lambdas(home_id, away_id)

        # Over/Under lines
        over_1_5 = float(1 - matrix[:2, :2].sum())  # at least 2 goals combined
        # More precise: sum where total >= 2
        over_1_5 = float(sum(
            matrix[i, j] for i in range(11) for j in range(11) if i + j >= 2
        ))
        over_2_5 = float(sum(
            matrix[i, j] for i in range(11) for j in range(11) if i + j >= 3
        ))
        over_3_5 = float(sum(
            matrix[i, j] for i in range(11) for j in range(11) if i + j >= 4
        ))

        return GoalsProbabilities(
            over_1_5=over_1_5,
            under_1_5=1.0 - over_1_5,
            over_2_5=over_2_5,
            under_2_5=1.0 - over_2_5,
            over_3_5=over_3_5,
            under_3_5=1.0 - over_3_5,
            expected_home=lam1,
            expected_away=lam2,
        )

    def predict_1x2(self, home_id: int, away_id: int) -> MatchResultProbabilities:
        matrix = self.predict_score_matrix(home_id, away_id)
        home_win = float(sum(matrix[i, j] for i in range(11) for j in range(11) if i > j))
        draw = float(sum(matrix[i, i] for i in range(11)))
        away_win = float(1.0 - home_win - draw)
        return MatchResultProbabilities(
            home_win=max(0.0, home_win),
            draw=max(0.0, draw),
            away_win=max(0.0, away_win),
        )

    def predict_btts(self, home_id: int, away_id: int) -> float:
        """P(both teams score >= 1)."""
        matrix = self.predict_score_matrix(home_id, away_id)
        p_home_blanks = float(matrix[:, 0].sum())   # away scores 0
        p_away_blanks = float(matrix[0, :].sum())   # home scores 0
        p_neither = float(matrix[0, 0])
        return max(0.0, 1.0 - p_home_blanks - p_away_blanks + p_neither)

    def team_ids(self) -> list[int]:
        return list(self._attack.keys())

    def get_params(self) -> bytes:
        return pickle.dumps({
            "attack": self._attack,
            "defense": self._defense,
            "home_adv": self._home_adv,
            "rho": self._rho,
        })

    @classmethod
    def from_params(cls, data: bytes) -> "DixonColesModel":
        model = cls()
        p = pickle.loads(data)
        model._attack = p["attack"]
        model._defense = p["defense"]
        model._home_adv = p["home_adv"]
        model._rho = p["rho"]
        model._is_fitted = True
        return model

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _lambdas(self, home_id: int, away_id: int) -> tuple[float, float]:
        """Expected goals for home and away teams."""
        # Fall back to league average (1.0) for unknown teams
        a_h = self._attack.get(home_id, 1.0)
        d_h = self._defense.get(home_id, 1.0)
        a_a = self._attack.get(away_id, 1.0)
        d_a = self._defense.get(away_id, 1.0)

        lam1 = a_h * d_a * self._home_adv
        lam2 = a_a * d_h
        return float(lam1), float(lam2)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")


# ---------------------------------------------------------------------------
# Optimisation objective (module-level for pickle compatibility)
# ---------------------------------------------------------------------------

def _neg_log_likelihood(
    params: np.ndarray,
    matches_df: pd.DataFrame,
    team_index: dict[int, int],
    n: int,
) -> float:
    alphas = params[:n]
    betas = params[n:2 * n]
    home_adv = params[2 * n]
    rho = params[2 * n + 1]

    # Vectorised approach
    home_idx = matches_df["home_team"].map(team_index).to_numpy()
    away_idx = matches_df["away_team"].map(team_index).to_numpy()
    hg = matches_df["home_goals"].to_numpy(dtype=int)
    ag = matches_df["away_goals"].to_numpy(dtype=int)
    weights = matches_df["weight"].to_numpy()

    lam1 = alphas[home_idx] * betas[away_idx] * home_adv
    lam2 = alphas[away_idx] * betas[home_idx]

    # Poisson log-likelihoods (vectorised)
    log_p1 = hg * np.log(np.maximum(lam1, 1e-10)) - lam1
    log_p2 = ag * np.log(np.maximum(lam2, 1e-10)) - lam2

    # Tau correction (only applies to low-score outcomes)
    tau = np.ones(len(matches_df), dtype=np.float64)
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)

    tau[m00] = np.maximum(1 - lam1[m00] * lam2[m00] * rho, 1e-10)
    tau[m10] = np.maximum(1 + lam2[m10] * rho, 1e-10)
    tau[m01] = np.maximum(1 + lam1[m01] * rho, 1e-10)
    tau[m11] = np.maximum(1 - rho, 1e-10)

    log_lik = weights * (np.log(tau) + log_p1 + log_p2)

    # Identifiability constraint: sum(attack) ≈ n
    penalty = 1000.0 * (np.sum(alphas) - n) ** 2

    return float(-np.sum(log_lik) + penalty)

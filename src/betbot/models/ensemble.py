"""Ensemble coordinator: runs all models for a given match."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from betbot.config import DC_BTTS_WEIGHT, LR_BTTS_WEIGHT
from betbot.data.repositories import (
    Match,
    MatchStats,
    ModelStateRepository,
    OddsRepository,
)
from betbot.markets.value import assign_confidence, calculate_edge
from betbot.models.dixon_coles import DixonColesModel
from betbot.models.features import (
    _team_avg_corners_against,
    _team_avg_corners_for,
    _team_avg_fouls,
    _team_avg_yellows,
    _team_conceded_rate,
    _team_scored_rate,
)
from betbot.models.logistic_btts import LogisticBTTSModel
from betbot.models.poisson_glm import PoissonCornersModel
from betbot.models.xgboost_cards import XGBoostCardsModel


@dataclass(frozen=True)
class MarketResult:
    market: str
    selection: str
    model_prob: float
    bookmaker_odds: float | None
    implied_prob: float | None
    edge: float | None
    confidence: str


@dataclass(frozen=True)
class MatchPrediction:
    match_id: int
    results: list[MarketResult]


class EnsemblePredictor:
    def __init__(
        self,
        dc_model: DixonColesModel,
        cards_model: XGBoostCardsModel | None,
        corners_model: PoissonCornersModel | None,
        btts_model: LogisticBTTSModel | None,
    ) -> None:
        self._dc = dc_model
        self._cards = cards_model
        self._corners = corners_model
        self._btts = btts_model

    @classmethod
    def from_db(cls, model_repo: ModelStateRepository) -> "EnsemblePredictor":
        """Load all available models from the database."""
        dc = _load_model(model_repo, DixonColesModel.MODEL_NAME, DixonColesModel)
        if dc is None or not dc.is_fitted:
            raise RuntimeError(
                "Dixon-Coles model not trained. Run: betbot train --model dixon-coles"
            )

        cards = _load_model(model_repo, XGBoostCardsModel.MODEL_NAME, XGBoostCardsModel)
        corners = _load_model(model_repo, PoissonCornersModel.MODEL_NAME, PoissonCornersModel)
        btts = _load_model(model_repo, LogisticBTTSModel.MODEL_NAME, LogisticBTTSModel)

        return cls(dc, cards, corners, btts)

    def predict(
        self,
        match: Match,
        recent_home: list[Any],
        recent_away: list[Any],
        odds_list: list[Any],
    ) -> MatchPrediction:
        results: list[MarketResult] = []
        odds_map = _build_odds_map(odds_list)

        # --- Goals O/U and 1X2 from Dixon-Coles ---
        goals = self._dc.predict_goals(match.home_team_id, match.away_team_id)
        result_1x2 = self._dc.predict_1x2(match.home_team_id, match.away_team_id)

        for selection, prob, market, odds_key in [
            ("over_2.5", goals.over_2_5, "goals_ou", "over_2.5"),
            ("under_2.5", goals.under_2_5, "goals_ou", "under_2.5"),
            ("over_1.5", goals.over_1_5, "goals_ou", "over_1.5"),
            ("under_1.5", goals.under_1_5, "goals_ou", "under_1.5"),
            ("over_3.5", goals.over_3_5, "goals_ou", "over_3.5"),
            ("under_3.5", goals.under_3_5, "goals_ou", "under_3.5"),
            ("home", result_1x2.home_win, "1x2", "home"),
            ("draw", result_1x2.draw, "1x2", "draw"),
            ("away", result_1x2.away_win, "1x2", "away"),
        ]:
            bk_odds = odds_map.get((market, odds_key))
            edge = calculate_edge(prob, bk_odds) if bk_odds else None
            results.append(MarketResult(
                market=market,
                selection=selection,
                model_prob=round(prob, 4),
                bookmaker_odds=bk_odds,
                implied_prob=round(1 / bk_odds, 4) if bk_odds else None,
                edge=round(edge, 4) if edge is not None else None,
                confidence=assign_confidence(edge) if edge is not None else "none",
            ))

        # --- BTTS ---
        dc_btts = self._dc.predict_btts(match.home_team_id, match.away_team_id)

        if self._btts and self._btts.is_fitted and recent_home and recent_away:
            home_matches = [m for m, _ in recent_home] if recent_home and isinstance(recent_home[0], tuple) else recent_home
            away_matches = [m for m, _ in recent_away] if recent_away and isinstance(recent_away[0], tuple) else recent_away

            lr_btts = self._btts.predict_proba(
                home_scored_rate=_team_scored_rate(match.home_team_id, home_matches[-10:]),
                home_conceded_rate=_team_conceded_rate(match.home_team_id, home_matches[-10:]),
                away_scored_rate=_team_scored_rate(match.away_team_id, away_matches[-10:]),
                away_conceded_rate=_team_conceded_rate(match.away_team_id, away_matches[-10:]),
            )
            btts_prob = DC_BTTS_WEIGHT * dc_btts + LR_BTTS_WEIGHT * lr_btts
        else:
            btts_prob = dc_btts

        bk_odds = odds_map.get(("btts", "yes"))
        edge = calculate_edge(btts_prob, bk_odds) if bk_odds else None
        results.append(MarketResult(
            market="btts",
            selection="yes",
            model_prob=round(btts_prob, 4),
            bookmaker_odds=bk_odds,
            implied_prob=round(1 / bk_odds, 4) if bk_odds else None,
            edge=round(edge, 4) if edge is not None else None,
            confidence=assign_confidence(edge) if edge is not None else "none",
        ))

        # --- Cards ---
        if self._cards and self._cards.is_fitted and recent_home and recent_away:
            home_hist = recent_home if not isinstance(recent_home[0], tuple) else [x for x in recent_home]
            away_hist = recent_away if not isinstance(recent_away[0], tuple) else [x for x in recent_away]

            # Extract (match, stats) tuples if available
            h_pairs = recent_home if isinstance(recent_home[0], tuple) else []
            a_pairs = recent_away if isinstance(recent_away[0], tuple) else []

            if h_pairs and a_pairs:
                cards_pred = self._cards.predict(
                    home_yellows_avg=_team_avg_yellows(match.home_team_id, h_pairs[-10:]),
                    away_yellows_avg=_team_avg_yellows(match.away_team_id, a_pairs[-10:]),
                    home_fouls_avg=_team_avg_fouls(match.home_team_id, h_pairs[-10:]),
                    away_fouls_avg=_team_avg_fouls(match.away_team_id, a_pairs[-10:]),
                )
                for selection, prob, odds_key in [
                    ("over_3.5", cards_pred.p_over_3_5, "over_3.5"),
                    ("over_4.5", cards_pred.p_over_4_5, "over_4.5"),
                    ("any_red", cards_pred.p_any_red, "any_red"),
                ]:
                    bk_odds = odds_map.get(("cards_ou", odds_key))
                    edge = calculate_edge(prob, bk_odds) if bk_odds else None
                    results.append(MarketResult(
                        market="cards_ou",
                        selection=selection,
                        model_prob=round(prob, 4),
                        bookmaker_odds=bk_odds,
                        implied_prob=round(1 / bk_odds, 4) if bk_odds else None,
                        edge=round(edge, 4) if edge is not None else None,
                        confidence=assign_confidence(edge) if edge is not None else "none",
                    ))

        # --- Corners ---
        if self._corners and self._corners.is_fitted:
            h_pairs = recent_home if isinstance(recent_home[0] if recent_home else None, tuple) else []
            a_pairs = recent_away if isinstance(recent_away[0] if recent_away else None, tuple) else []

            if h_pairs and a_pairs:
                corners_pred = self._corners.predict(
                    home_corners_for=_team_avg_corners_for(match.home_team_id, h_pairs[-10:]),
                    home_corners_against=_team_avg_corners_against(match.home_team_id, h_pairs[-10:]),
                    away_corners_for=_team_avg_corners_for(match.away_team_id, a_pairs[-10:]),
                    away_corners_against=_team_avg_corners_against(match.away_team_id, a_pairs[-10:]),
                )
                for selection, prob, odds_key in [
                    ("over_9.5", corners_pred.p_over_9_5, "over_9.5"),
                    ("over_10.5", corners_pred.p_over_10_5, "over_10.5"),
                    ("over_11.5", corners_pred.p_over_11_5, "over_11.5"),
                ]:
                    bk_odds = odds_map.get(("corners_ou", odds_key))
                    edge = calculate_edge(prob, bk_odds) if bk_odds else None
                    results.append(MarketResult(
                        market="corners_ou",
                        selection=selection,
                        model_prob=round(prob, 4),
                        bookmaker_odds=bk_odds,
                        implied_prob=round(1 / bk_odds, 4) if bk_odds else None,
                        edge=round(edge, 4) if edge is not None else None,
                        confidence=assign_confidence(edge) if edge is not None else "none",
                    ))

        return MatchPrediction(match_id=match.id, results=results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_odds_map(odds_list: list[Any]) -> dict[tuple[str, str], float]:
    return {(o.market, o.selection): o.odds_decimal for o in odds_list}


def _load_model(repo: ModelStateRepository, name: str, cls: type) -> Any:
    result = repo.load(name)
    if result is None:
        return None
    params_bytes, _, _ = result
    try:
        return cls.from_params(params_bytes)
    except Exception:
        return None

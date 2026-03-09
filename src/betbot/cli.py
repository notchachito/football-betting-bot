"""betbot CLI — entry point for all commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from betbot.config import (
    CURRENT_SEASON,
    DAILY_CALL_LIMIT,
    LEAGUE_IDS,
    Settings,
    resolve_league_name,
)
from betbot.data.api_client import ApiFootballClient, BudgetExhaustedError
from betbot.data.cache import ResponseCache
from betbot.data.db import get_connection, init_db
from betbot.data.repositories import (
    ApiCallRepository,
    LeagueRepository,
    MatchRepository,
    ModelStateRepository,
    OddsRepository,
    PredictionRepository,
    StatsRepository,
    TeamRepository,
)
from betbot.data.sync import SyncOrchestrator
from betbot.display.tables import (
    console,
    render_budget,
    render_match_predictions,
    render_sync_report,
    render_training_summary,
)
from betbot.models.dixon_coles import DixonColesModel
from betbot.models.ensemble import EnsemblePredictor
from betbot.models.features import (
    build_btts_features,
    build_cards_features,
    build_corners_features,
    build_dixon_coles_dataset,
)
from betbot.models.logistic_btts import LogisticBTTSModel
from betbot.models.poisson_glm import PoissonCornersModel
from betbot.models.xgboost_cards import XGBoostCardsModel


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def main() -> None:
    """Football betting analytics bot — top 5 leagues + Champions League."""
    pass


# ---------------------------------------------------------------------------
# betbot sync
# ---------------------------------------------------------------------------

@main.command()
@click.option("--league", "-l", default=None, help="League name or alias (default: all)")
@click.option("--force", is_flag=True, help="Ignore cache TTL and force re-fetch")
def sync(league: str | None, force: bool) -> None:
    """Fetch latest fixtures, results, statistics and odds."""
    try:
        settings = Settings.from_env()
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/]")
        sys.exit(1)

    league_name: str | None = None
    if league:
        try:
            league_name = resolve_league_name(league)
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            sys.exit(1)

    conn = get_connection(settings.db_path)
    init_db(conn)

    cache = ResponseCache(settings.cache_dir)
    api_calls = ApiCallRepository(conn)
    client = ApiFootballClient(settings, cache, api_calls)

    orchestrator = SyncOrchestrator(
        client=client,
        league_repo=LeagueRepository(conn),
        team_repo=TeamRepository(conn),
        match_repo=MatchRepository(conn),
        stats_repo=StatsRepository(conn),
        odds_repo=OddsRepository(conn),
        api_calls_repo=api_calls,
    )

    label = league_name or "all leagues"
    with console.status(f"[bold]Syncing {label}...[/]"):
        try:
            report = orchestrator.sync_all(league_name)
        except BudgetExhaustedError as e:
            console.print(f"[red]Budget exhausted: {e}[/]")
            sys.exit(1)

    render_sync_report(
        report.fixtures_synced,
        report.stats_synced,
        report.odds_synced,
        report.calls_used,
        report.calls_remaining,
        report.errors,
    )
    client.close()
    conn.close()


# ---------------------------------------------------------------------------
# betbot train
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--model", "-m",
    default="all",
    type=click.Choice(["all", "dixon-coles", "xgboost-cards", "poisson-corners", "logistic-btts"]),
    help="Which model to train",
)
@click.option("--league", "-l", default=None, help="Limit training data to one league")
def train(model: str, league: str | None) -> None:
    """Train or retrain prediction models on stored historical data."""
    try:
        settings = Settings.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    conn = get_connection(settings.db_path)
    init_db(conn)

    match_repo = MatchRepository(conn)
    stats_repo = StatsRepository(conn)
    model_repo = ModelStateRepository(conn)

    league_id: int | None = None
    if league:
        try:
            name = resolve_league_name(league)
            league_id = LEAGUE_IDS[name]
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            sys.exit(1)

    # Gather training data
    all_matches = []
    all_stats = []
    target_leagues = [league_id] if league_id else list(LEAGUE_IDS.values())

    for lid in target_leagues:
        matches = match_repo.get_finished(lid, CURRENT_SEASON)
        all_matches.extend(matches)
        match_ids = [m.id for m in matches]
        if match_ids:
            placeholders = ",".join("?" * len(match_ids))
            rows = conn.execute(
                f"SELECT * FROM match_stats WHERE match_id IN ({placeholders})", match_ids
            ).fetchall()
            from betbot.data.repositories import MatchStats
            all_stats.extend([MatchStats(**dict(r)) for r in rows])

    console.print(f"[dim]Training on {len(all_matches)} matches, {len(all_stats)} with stats[/]")

    train_dc = model in ("all", "dixon-coles")
    train_cards = model in ("all", "xgboost-cards")
    train_corners = model in ("all", "poisson-corners")
    train_btts_lr = model in ("all", "logistic-btts")

    # Dixon-Coles
    if train_dc:
        with console.status("[bold]Fitting Dixon-Coles...[/]"):
            try:
                df = build_dixon_coles_dataset(all_matches)
                dc = DixonColesModel().fit(df)
                model_repo.save(
                    DixonColesModel.MODEL_NAME,
                    dc.get_params(),
                    {"matches": len(df), "teams": len(dc.team_ids())},
                    CURRENT_SEASON,
                )
                render_training_summary("Dixon-Coles", {"matches": len(df), "teams": len(dc.team_ids())})
            except ValueError as e:
                console.print(f"[yellow]⚠ Dixon-Coles skipped: {e}[/]")

    # XGBoost Cards
    if train_cards:
        with console.status("[bold]Training XGBoost cards model...[/]"):
            try:
                df = build_cards_features(all_matches, all_stats)
                cards = XGBoostCardsModel().fit(df)
                model_repo.save(
                    XGBoostCardsModel.MODEL_NAME,
                    cards.get_params(),
                    {"samples": len(df)},
                    CURRENT_SEASON,
                )
                render_training_summary("XGBoost Cards", {"samples": len(df)})
            except ValueError as e:
                console.print(f"[yellow]⚠ Cards model skipped: {e}[/]")

    # Poisson Corners
    if train_corners:
        with console.status("[bold]Training Poisson corners model...[/]"):
            try:
                df = build_corners_features(all_matches, all_stats)
                corners = PoissonCornersModel().fit(df)
                model_repo.save(
                    PoissonCornersModel.MODEL_NAME,
                    corners.get_params(),
                    {"samples": len(df)},
                    CURRENT_SEASON,
                )
                render_training_summary("Poisson Corners", {"samples": len(df)})
            except ValueError as e:
                console.print(f"[yellow]⚠ Corners model skipped: {e}[/]")

    # Logistic BTTS
    if train_btts_lr:
        with console.status("[bold]Training logistic BTTS model...[/]"):
            try:
                df = build_btts_features(all_matches)
                btts_model = LogisticBTTSModel().fit(df)
                model_repo.save(
                    LogisticBTTSModel.MODEL_NAME,
                    btts_model.get_params(),
                    {"samples": len(df)},
                    CURRENT_SEASON,
                )
                render_training_summary("Logistic BTTS", {"samples": len(df)})
            except ValueError as e:
                console.print(f"[yellow]⚠ BTTS model skipped: {e}[/]")

    conn.close()


# ---------------------------------------------------------------------------
# betbot predict
# ---------------------------------------------------------------------------

@main.command()
@click.option("--league", "-l", default=None, help="League name or alias")
@click.option("--date", "-d", default="today", help="Date: today | tomorrow | YYYY-MM-DD")
@click.option(
    "--market", "-m", default="all",
    type=click.Choice(["all", "goals", "1x2", "btts", "cards", "corners"]),
)
@click.option("--min-edge", default=0.0, type=float, help="Min edge to show (e.g. 0.05)")
def predict(league: str | None, date: str, market: str, min_edge: float) -> None:
    """Show predictions for upcoming matches."""
    try:
        settings = Settings.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    conn = get_connection(settings.db_path)
    init_db(conn)

    match_repo = MatchRepository(conn)
    stats_repo = StatsRepository(conn)
    odds_repo = OddsRepository(conn)
    team_repo = TeamRepository(conn)
    model_repo = ModelStateRepository(conn)

    league_id: int | None = None
    league_label = "All Leagues"
    if league:
        try:
            name = resolve_league_name(league)
            league_id = LEAGUE_IDS[name]
            league_label = name
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            sys.exit(1)

    try:
        predictor = EnsemblePredictor.from_db(model_repo)
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    upcoming = match_repo.get_upcoming(league_id, days_ahead=2)
    if not upcoming:
        console.print(f"[yellow]No upcoming matches found for {league_label}.[/]")
        console.print("[dim]Try: betbot sync[/]")
        conn.close()
        return

    market_filter = {
        "goals": "goals_ou", "1x2": "1x2", "btts": "btts",
        "cards": "cards_ou", "corners": "corners_ou",
    }.get(market)

    console.print(f"\n[bold cyan]Football Analytics Bot[/] · {league_label}\n")

    for match in upcoming:
        home_team = team_repo.get_by_id(match.home_team_id)
        away_team = team_repo.get_by_id(match.away_team_id)
        home_name = home_team.name if home_team else str(match.home_team_id)
        away_name = away_team.name if away_team else str(match.away_team_id)

        # Build recent history with stats
        home_matches = match_repo.get_team_recent(match.home_team_id, 15, match.match_date)
        away_matches = match_repo.get_team_recent(match.away_team_id, 15, match.match_date)

        home_match_ids = [m.id for m in home_matches]
        away_match_ids = [m.id for m in away_matches]

        home_stats = {s.match_id: s for s in stats_repo.get_for_team(match.home_team_id, home_match_ids)}
        away_stats = {s.match_id: s for s in stats_repo.get_for_team(match.away_team_id, away_match_ids)}

        home_pairs = [(m, home_stats[m.id]) for m in home_matches if m.id in home_stats]
        away_pairs = [(m, away_stats[m.id]) for m in away_matches if m.id in away_stats]

        odds_list = odds_repo.get_by_match(match.id)

        try:
            prediction = predictor.predict(match, home_pairs, away_pairs, odds_list)
        except Exception as e:
            console.print(f"[red]Error predicting {home_name} vs {away_name}: {e}[/]")
            continue

        # Resolve league name from DB
        league_row = conn.execute(
            "SELECT name FROM leagues WHERE id=?", (match.league_id,)
        ).fetchone()
        match_league = league_row["name"] if league_row else str(match.league_id)

        render_match_predictions(
            prediction,
            home_name,
            away_name,
            match.match_date,
            match_league,
            market_filter=market_filter,
            min_edge=min_edge,
        )

    conn.close()


# ---------------------------------------------------------------------------
# betbot budget
# ---------------------------------------------------------------------------

@main.command()
def budget() -> None:
    """Show today's API call budget usage."""
    try:
        settings = Settings.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    conn = get_connection(settings.db_path)
    init_db(conn)
    api_calls = ApiCallRepository(conn)

    used = api_calls.today_count()
    remaining = api_calls.remaining()
    render_budget(used, remaining, DAILY_CALL_LIMIT)
    conn.close()


# ---------------------------------------------------------------------------
# betbot status
# ---------------------------------------------------------------------------

@main.command()
def status() -> None:
    """Show data freshness and model training dates."""
    try:
        settings = Settings.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    conn = get_connection(settings.db_path)
    init_db(conn)
    model_repo = ModelStateRepository(conn)

    from rich.table import Table as RTable
    from rich import box as rbox

    # Leagues
    table = RTable(title="League Data", box=rbox.SIMPLE_HEAD)
    table.add_column("League")
    table.add_column("Total Matches", justify="right")
    table.add_column("Finished", justify="right")
    table.add_column("Upcoming", justify="right")
    table.add_column("With Stats", justify="right")

    for name, lid in LEAGUE_IDS.items():
        total = conn.execute(
            "SELECT COUNT(*) as c FROM matches WHERE league_id=?", (lid,)
        ).fetchone()["c"]
        finished = conn.execute(
            "SELECT COUNT(*) as c FROM matches WHERE league_id=? AND status='FT'", (lid,)
        ).fetchone()["c"]
        upcoming = conn.execute(
            "SELECT COUNT(*) as c FROM matches WHERE league_id=? AND status='NS'", (lid,)
        ).fetchone()["c"]
        with_stats = conn.execute(
            """SELECT COUNT(*) as c FROM matches m
               JOIN match_stats s ON s.match_id=m.id
               WHERE m.league_id=?""", (lid,)
        ).fetchone()["c"]
        table.add_row(name, str(total), str(finished), str(upcoming), str(with_stats))

    console.print(table)

    # Models
    mtable = RTable(title="Model Status", box=rbox.SIMPLE_HEAD)
    mtable.add_column("Model")
    mtable.add_column("Trained At")
    mtable.add_column("Status")

    for model_name in ["dixon_coles", "xgboost_cards", "poisson_corners", "logistic_btts"]:
        trained_at = model_repo.get_training_date(model_name)
        if trained_at:
            mtable.add_row(model_name, trained_at, "[green]✓ Ready[/]")
        else:
            mtable.add_row(model_name, "—", "[red]Not trained[/]")

    console.print(mtable)
    conn.close()


if __name__ == "__main__":
    main()

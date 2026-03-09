"""Integration tests for repositories against real SQLite."""

from betbot.data.repositories import (
    LeagueRepository,
    MatchRepository,
    StatsRepository,
    TeamRepository,
)


def test_league_upsert_and_get(db_conn, sample_leagues):
    repo = LeagueRepository(db_conn)
    for league in sample_leagues:
        repo.upsert(league)
    result = repo.get_all(season=2024)
    assert len(result) == 2
    names = {r.name for r in result}
    assert "Premier League" in names


def test_team_upsert_and_get(db_conn, sample_leagues, sample_teams):
    league_repo = LeagueRepository(db_conn)
    for league in sample_leagues:
        league_repo.upsert(league)

    team_repo = TeamRepository(db_conn)
    for team in sample_teams:
        team_repo.upsert(team)

    teams = team_repo.get_by_league(39, 2024)
    assert len(teams) == 4


def test_match_upsert_and_get(db_conn, sample_leagues, sample_teams, sample_matches):
    league_repo = LeagueRepository(db_conn)
    for league in sample_leagues:
        league_repo.upsert(league)

    team_repo = TeamRepository(db_conn)
    for team in sample_teams:
        team_repo.upsert(team)

    match_repo = MatchRepository(db_conn)
    match_repo.upsert_many(sample_matches)

    finished = match_repo.get_finished(39, 2024)
    assert len(finished) == len(sample_matches)


def test_match_get_team_recent(db_conn, sample_leagues, sample_teams, sample_matches):
    league_repo = LeagueRepository(db_conn)
    for league in sample_leagues:
        league_repo.upsert(league)

    team_repo = TeamRepository(db_conn)
    for team in sample_teams:
        team_repo.upsert(team)

    match_repo = MatchRepository(db_conn)
    match_repo.upsert_many(sample_matches)

    recent = match_repo.get_team_recent(33, n=5)
    assert len(recent) <= 5
    for m in recent:
        assert m.home_team_id == 33 or m.away_team_id == 33


def test_stats_upsert_and_get(db_conn, sample_leagues, sample_teams, sample_matches, sample_stats):
    league_repo = LeagueRepository(db_conn)
    for league in sample_leagues:
        league_repo.upsert(league)

    team_repo = TeamRepository(db_conn)
    for team in sample_teams:
        team_repo.upsert(team)

    match_repo = MatchRepository(db_conn)
    match_repo.upsert_many(sample_matches)

    stats_repo = StatsRepository(db_conn)
    for s in sample_stats:
        stats_repo.upsert(s)

    s = stats_repo.get_by_match(sample_matches[0].id)
    assert s is not None
    assert s.match_id == sample_matches[0].id


def test_missing_stats_ids(db_conn, sample_leagues, sample_teams, sample_matches):
    league_repo = LeagueRepository(db_conn)
    for league in sample_leagues:
        league_repo.upsert(league)

    team_repo = TeamRepository(db_conn)
    for team in sample_teams:
        team_repo.upsert(team)

    match_repo = MatchRepository(db_conn)
    match_repo.upsert_many(sample_matches)

    # No stats inserted — all finished matches should be missing
    missing = match_repo.get_missing_stats_ids()
    assert len(missing) == len(sample_matches)

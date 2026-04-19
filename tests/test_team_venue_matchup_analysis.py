"""Tests for ``team_venue_matchup_analysis`` CLI helpers."""

from __future__ import annotations

import pytest

import pandas as pd

from src.scripts.team_venue_matchup_analysis import (
    TeamVenueAnalysisRequest,
    _h2h_batter_aggregate_vs_pool,
    _h2h_bowler_aggregate_vs_pool,
    _h2h_rows_meeting_min_balls,
    _resolve_franchise_venue,
)


def test_resolved_team_names_mi_pbks() -> None:
    """Short codes map to full franchise names."""

    req = TeamVenueAnalysisRequest(
        team_a_short="mi", team_b_short="pbks", venue_query="Wankhede"
    )
    assert req.resolved_team_names() == ("Mumbai Indians", "Punjab Kings")


def test_resolved_team_names_unknown() -> None:
    """Invalid short code raises ``ValueError``."""

    req = TeamVenueAnalysisRequest(
        team_a_short="XX", team_b_short="MI", venue_query="Eden"
    )
    with pytest.raises(ValueError, match="Unknown team"):
        req.resolved_team_names()


def test_resolve_franchise_venue_substring() -> None:
    """User substring resolves to a single catalog venue."""

    catalog = ["Eden Gardens", "Wankhede Stadium", "Other Ground"]
    assert _resolve_franchise_venue("Wankhede", catalog) == "Wankhede Stadium"


def test_resolve_franchise_venue_ambiguous() -> None:
    """Multiple matches raise with candidate list."""

    catalog = ["Stadium A West", "Stadium B West"]
    with pytest.raises(ValueError, match="Ambiguous"):
        _resolve_franchise_venue("West", catalog)


def test_h2h_rows_meeting_min_balls_filters_and_sorts_sr() -> None:
    """H2H helper drops rows below min balls and sorts SR ascending."""

    df = pd.DataFrame(
        {
            "batter": ["A", "B", "C"],
            "bowler": ["X", "Y", "Z"],
            "balls": [10, 5, 12],
            "runs": [10, 25, 30],
            "dismissals": [2, 0, 3],
            "matches": [4, 1, 2],
            "strike_rate": [100.0, 500.0, 250.0],
        }
    )
    out = _h2h_rows_meeting_min_balls(df, min_balls=10)
    assert list(out["batter"]) == ["A", "C"]
    assert len(out) == 2
    row_a = out.loc[out["batter"] == "A"].iloc[0]
    assert abs(float(row_a["bowler_effectiveness"]) - 0.5) < 1e-9  # 2 dismissals / 4 matches
    row_c = out.loc[out["batter"] == "C"].iloc[0]
    assert abs(float(row_c["bowler_effectiveness"]) - 1.5) < 1e-9  # 3 / 2 matches


def test_h2h_bowler_aggregate_vs_pool_wickets_economy_sort() -> None:
    """Bowler pool aggregate: wickets, runs, economy; min balls bowled; sort wickets."""

    df = pd.DataFrame(
        {
            "batter": ["A1", "A2", "B1", "B2"],
            "bowler": ["X", "X", "Y", "Y"],
            "balls": [6, 6, 6, 6],
            "runs": [12, 12, 20, 16],
            "dismissals": [1, 1, 1, 0],
            "strike_rate": [200.0, 200.0, 200.0, 200.0],
        }
    )
    out = _h2h_bowler_aggregate_vs_pool(df, min_balls=12)
    assert list(out["bowler"]) == ["X", "Y"]
    row_x = out.loc[out["bowler"] == "X"].iloc[0]
    assert int(row_x["wickets"]) == 2
    assert int(row_x["runs_conceded"]) == 24
    assert int(row_x["balls_bowled"]) == 12
    assert abs(float(row_x["economy"]) - 12.0) < 1e-6
    row_y = out.loc[out["bowler"] == "Y"].iloc[0]
    assert int(row_y["wickets"]) == 1
    assert int(row_y["runs_conceded"]) == 36
    assert abs(float(row_y["economy"]) - 18.0) < 1e-6


def test_h2h_batter_aggregate_vs_pool_totals_and_min_balls() -> None:
    """Pool aggregate sums runs, balls, dismissals; filters by total balls."""

    df = pd.DataFrame(
        {
            "batter": ["A", "A", "B", "B"],
            "bowler": ["X", "Y", "Z", "W"],
            "balls": [4, 4, 3, 2],
            "runs": [8, 4, 9, 4],
            "dismissals": [0, 1, 0, 0],
            "strike_rate": [200.0, 100.0, 300.0, 200.0],
        }
    )
    out = _h2h_batter_aggregate_vs_pool(df, min_balls=8)
    row_a = out.loc[out["batter"] == "A"].iloc[0]
    assert int(row_a["total_runs"]) == 12
    assert int(row_a["dismissals"]) == 1
    assert int(row_a["balls_faced"]) == 8
    assert abs(float(row_a["strike_rate"]) - 150.0) < 1e-6
    assert len(out) == 1


def test_h2h_rows_meeting_min_balls_rejects_zero() -> None:
    """Minimum balls must be at least one."""

    df = pd.DataFrame(
        {
            "batter": ["A"],
            "bowler": ["X"],
            "balls": [5],
            "dismissals": [0],
            "matches": [1],
            "strike_rate": [100.0],
        }
    )
    with pytest.raises(ValueError, match="at least 1"):
        _h2h_rows_meeting_min_balls(df, min_balls=0)

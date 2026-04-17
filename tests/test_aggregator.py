from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.aggregator import aggregate_all


def make_delivery(**kwargs: object) -> dict[str, object]:
    row: dict[str, object] = {
        "match_id": "m1",
        "season": np.int64(2024),
        "date": "2024-01-01",
        "venue": "V1",
        "city": "C1",
        "team1": "A",
        "team2": "B",
        "toss_winner": "A",
        "toss_decision": "bat",
        "match_winner": "A",
        "win_by_runs": None,
        "win_by_wickets": np.int64(0),
        "player_of_match": "POM",
        "is_playoff": False,
        "match_number": np.int64(1),
        "innings_num": np.int64(1),
        "is_super_over": False,
        "batting_team": "A",
        "bowling_team": "B",
        "innings_total": np.int64(0),
        "innings_wickets": np.int64(0),
        "target": np.nan,
        "over": np.int64(0),
        "over_1indexed": np.int64(1),
        "ball": np.int64(1),
        "phase": "powerplay",
        "is_legal_delivery": True,
        "batter": "P1",
        "non_striker": "NS",
        "runs_batter": np.int64(0),
        "is_four": False,
        "is_six": False,
        "is_dot_ball": True,
        "bowler": "P2",
        "runs_bowler": np.int64(0),
        "is_wide": False,
        "is_noball": False,
        "wide_runs": np.int64(0),
        "noball_runs": np.int64(0),
        "bye_runs": np.int64(0),
        "legbye_runs": np.int64(0),
        "extras_total": np.int64(0),
        "extras_type": np.nan,
        "is_wicket": False,
        "wicket_player_out": np.nan,
        "wicket_kind": np.nan,
        "wicket_fielder1": np.nan,
        "wicket_fielder2": None,
        "fielder1_is_sub": None,
        "fielder2_is_sub": None,
        "is_bowler_wicket": False,
        "wicket2_player_out": None,
        "wicket2_kind": None,
        "wicket2_fielder1": None,
        "wicket2_fielder2": None,
        "wicket2_fielder1_is_sub": None,
        "wicket2_fielder2_is_sub": None,
    }
    row.update(kwargs)
    return row


def _write_match(tmp_path: Path, match_id: str, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df["match_id"] = match_id
    (tmp_path / f"{match_id}.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp_path / f"{match_id}.parquet", index=False)


def test_career_batting_runs_sr_average_two_matches(tmp_path: Path) -> None:
    processed = tmp_path / "matches"
    out = tmp_path / "agg"
    processed.mkdir()

    m1 = [make_delivery(match_id="m1", batter="P1", runs_batter=np.int64(1)) for _ in range(10)]
    m1.append(
        make_delivery(match_id="m1", batter="P1", is_wicket=True, wicket_player_out="P1", wicket_kind="caught", is_bowler_wicket=True)
    )
    _write_match(processed, "m1", m1)

    m2 = [make_delivery(match_id="m2", date="2025-01-01", season=np.int64(2025), batter="P1", runs_batter=np.int64(2), runs_bowler=np.int64(2)) for _ in range(5)]
    _write_match(processed, "m2", m2)

    res = aggregate_all(str(processed), str(out), force=True)
    bat = res["career_batting"].loc[res["career_batting"]["player"] == "P1"].iloc[0]
    assert int(bat["runs_total"]) == 20
    assert int(bat["balls_faced"]) == 16
    assert int(bat["dismissals"]) == 1
    assert float(bat["average"]) == 20.0
    assert float(bat["strike_rate"]) == 125.0


def test_career_bowling_wickets_economy_maidens_two_matches(tmp_path: Path) -> None:
    processed = tmp_path / "matches"
    out = tmp_path / "agg"
    processed.mkdir()

    m1 = []
    for ball in range(1, 7):
        m1.append(make_delivery(match_id="m1", bowler="P2", over_1indexed=np.int64(1), ball=np.int64(ball), runs_batter=np.int64(0), runs_bowler=np.int64(0)))
    m1.append(make_delivery(match_id="m1", bowler="P2", over_1indexed=np.int64(2), ball=np.int64(1), runs_batter=np.int64(1), runs_bowler=np.int64(1), is_wicket=True, wicket_kind="bowled", wicket_player_out="X", is_bowler_wicket=True))
    _write_match(processed, "m1", m1)

    m2 = [make_delivery(match_id="m2", date="2025-01-01", season=np.int64(2025), bowler="P2", over_1indexed=np.int64(1), ball=np.int64(1), runs_batter=np.int64(4), runs_bowler=np.int64(4), is_wicket=True, wicket_kind="lbw", wicket_player_out="Y", is_bowler_wicket=True)]
    _write_match(processed, "m2", m2)

    res = aggregate_all(str(processed), str(out), force=True)
    bowl = res["career_bowling"].loc[res["career_bowling"]["player"] == "P2"].iloc[0]
    assert int(bowl["wickets"]) == 2
    assert int(bowl["maidens"]) == 1
    assert float(bowl["overs_bowled"]) == (8 / 6.0)
    assert float(bowl["economy"]) == (5 / (8 / 6.0))


def test_career_fielding_catches_stumpings_summed(tmp_path: Path) -> None:
    processed = tmp_path / "matches"
    out = tmp_path / "agg"
    processed.mkdir()

    m1 = [
        make_delivery(match_id="m1", is_wicket=True, wicket_kind="caught", wicket_fielder1="P3", fielder1_is_sub=None),
        make_delivery(match_id="m1", is_wicket=True, wicket_kind="stumped", wicket_fielder1="P3", fielder1_is_sub=None),
    ]
    _write_match(processed, "m1", m1)

    res = aggregate_all(str(processed), str(out), force=True)
    f = res["career_fielding"].loc[res["career_fielding"]["player"] == "P3"].iloc[0]
    assert int(f["catches"]) == 1
    assert int(f["stumpings"]) == 1


def test_venue_splits_two_venues_correct_split(tmp_path: Path) -> None:
    processed = tmp_path / "matches"
    out = tmp_path / "agg"
    processed.mkdir()

    m1 = [make_delivery(match_id="m1", venue="V1", batter="P1", runs_batter=np.int64(1)) for _ in range(10)]
    _write_match(processed, "m1", m1)
    m2 = [make_delivery(match_id="m2", venue="V2", batter="P1", runs_batter=np.int64(2), runs_bowler=np.int64(2), date="2025-01-01", season=np.int64(2025)) for _ in range(5)]
    _write_match(processed, "m2", m2)

    res = aggregate_all(str(processed), str(out), force=True)
    vs = res["venue_splits"]
    v1 = vs.loc[(vs["player"] == "P1") & (vs["venue"] == "V1")].iloc[0]
    v2 = vs.loc[(vs["player"] == "P1") & (vs["venue"] == "V2")].iloc[0]
    assert int(v1["runs_total"]) == 10
    assert int(v2["runs_total"]) == 10


def test_venue_normalization_merges_trailing_city_suffix(tmp_path: Path) -> None:
    processed = tmp_path / "matches"
    out = tmp_path / "agg"
    processed.mkdir()

    m1 = [
        make_delivery(
            match_id="m1",
            venue="Wankhede Stadium, Mumbai",
            batter="P1",
            runs_batter=np.int64(1),
        )
        for _ in range(5)
    ]
    _write_match(processed, "m1", m1)
    m2 = [
        make_delivery(
            match_id="m2",
            venue="Wankhede Stadium",
            batter="P1",
            runs_batter=np.int64(1),
            date="2025-01-01",
            season=np.int64(2025),
        )
        for _ in range(3)
    ]
    _write_match(processed, "m2", m2)

    res = aggregate_all(str(processed), str(out), force=True)
    sub = res["venue_splits"].loc[res["venue_splits"]["player"] == "P1"]
    assert len(sub) == 1
    assert str(sub.iloc[0]["venue"]) == "Wankhede Stadium"
    assert int(sub.iloc[0]["matches"]) == 2


def test_phase_splits_powerplay_vs_death_sr(tmp_path: Path) -> None:
    processed = tmp_path / "matches"
    out = tmp_path / "agg"
    processed.mkdir()

    rows = []
    for i in range(10):
        rows.append(make_delivery(match_id="m1", over_1indexed=np.int64(1), ball=np.int64(i % 6 + 1), batter="P1", runs_batter=np.int64(1), runs_bowler=np.int64(1)))
    for i in range(5):
        rows.append(make_delivery(match_id="m1", over_1indexed=np.int64(16), ball=np.int64(i % 6 + 1), batter="P1", runs_batter=np.int64(2), runs_bowler=np.int64(2)))
    _write_match(processed, "m1", rows)

    res = aggregate_all(str(processed), str(out), force=True)
    ps = res["phase_splits"]
    pp = ps.loc[(ps["player"] == "P1") & (ps["phase"] == "powerplay")].iloc[0]
    death = ps.loc[(ps["player"] == "P1") & (ps["phase"] == "death")].iloc[0]
    assert float(pp["strike_rate"]) == 100.0
    assert float(death["strike_rate"]) == 200.0


def test_season_trends_two_seasons_avg(tmp_path: Path) -> None:
    processed = tmp_path / "matches"
    out = tmp_path / "agg"
    processed.mkdir()

    _write_match(processed, "m1", [make_delivery(match_id="m1", season=np.int64(2024), date="2024-01-01", batter="P1", runs_batter=np.int64(10), runs_bowler=np.int64(10))])
    _write_match(processed, "m2", [make_delivery(match_id="m2", season=np.int64(2025), date="2025-01-01", batter="P1", runs_batter=np.int64(0), runs_bowler=np.int64(0), is_wicket=True, wicket_player_out="P1", wicket_kind="caught", is_bowler_wicket=True)])

    res = aggregate_all(str(processed), str(out), force=True)
    st = res["season_trends"]
    s1 = st.loc[(st["player"] == "P1") & (st["season"] == 2024)].iloc[0]
    s2 = st.loc[(st["player"] == "P1") & (st["season"] == 2025)].iloc[0]
    assert float(s1["fantasy_total_avg"]) > float(s2["fantasy_total_avg"])


def test_team_latest_reflects_most_recent_by_date(tmp_path: Path) -> None:
    processed = tmp_path / "matches"
    out = tmp_path / "agg"
    processed.mkdir()

    _write_match(processed, "m1", [make_delivery(match_id="m1", season=np.int64(2024), date="2024-01-01", batter="P1", batting_team="A", bowling_team="B", runs_batter=np.int64(1))])
    _write_match(processed, "m2", [make_delivery(match_id="m2", season=np.int64(2025), date="2025-01-01", batter="P1", batting_team="C", bowling_team="D", runs_batter=np.int64(1))])

    res = aggregate_all(str(processed), str(out), force=True)
    bat = res["career_batting"].loc[res["career_batting"]["player"] == "P1"].iloc[0]
    assert str(bat["team_latest"]) == "C"


def test_retired_player_excluded_when_not_in_latest_season(tmp_path: Path) -> None:
    processed = tmp_path / "matches"
    out = tmp_path / "agg"
    processed.mkdir()

    _write_match(
        processed,
        "m1",
        [
            make_delivery(match_id="m1", batter="RETIRED", season=np.int64(2024), runs_batter=np.int64(1))
            for _ in range(5)
        ],
    )
    _write_match(
        processed,
        "m2",
        [
            make_delivery(
                match_id="m2",
                batter="ACTIVE",
                season=np.int64(2025),
                date="2025-02-01",
                runs_batter=np.int64(1),
            )
            for _ in range(5)
        ],
    )

    res = aggregate_all(str(processed), str(out), force=True, active_latest_season_only=True)
    names = set(res["career_batting"]["player"].astype(str).tolist())
    assert "ACTIVE" in names
    assert "RETIRED" not in names

    out2 = tmp_path / "agg_all"
    res_all = aggregate_all(str(processed), str(out2), force=True, active_latest_season_only=False)
    names_all = set(res_all["career_batting"]["player"].astype(str).tolist())
    assert "RETIRED" in names_all
    assert "ACTIVE" in names_all


def test_incremental_skip_when_outputs_exist(tmp_path: Path) -> None:
    processed = tmp_path / "matches"
    out = tmp_path / "agg"
    processed.mkdir()
    _write_match(processed, "m1", [make_delivery(match_id="m1", batter="P1", runs_batter=np.int64(1))])

    first = aggregate_all(str(processed), str(out), force=True)
    assert (out / "career_batting.csv").exists()

    # Add a new match, but run with force=False; should not change loaded results
    _write_match(processed, "m2", [make_delivery(match_id="m2", batter="P1", runs_batter=np.int64(100), date="2025-01-01", season=np.int64(2025))])
    second = aggregate_all(str(processed), str(out), force=False)
    assert int(second["career_batting"].loc[second["career_batting"]["player"] == "P1", "runs_total"].iloc[0]) == int(
        first["career_batting"].loc[first["career_batting"]["player"] == "P1", "runs_total"].iloc[0]
    )


def test_aggregate_all_empty_matches_dir_no_keyerror(tmp_path: Path) -> None:
    """No parquets must not raise (regression: duplicate team_latest on merge)."""

    processed = tmp_path / "matches"
    processed.mkdir()
    out = tmp_path / "agg"
    res = aggregate_all(str(processed), str(out), force=True)
    assert res["career_batting"].empty
    assert "team_latest" in res["career_batting"].columns
    assert res["career_bowling"].empty
    assert "team_latest" in res["career_bowling"].columns


def test_aggregate_all_reads_parquets_under_processed_matches(tmp_path: Path) -> None:
    """Processed root with only ``matches/*.parquet`` (parser layout) is supported."""

    root = tmp_path / "processed"
    matches = root / "matches"
    matches.mkdir(parents=True)
    _write_match(matches, "m1", [make_delivery(match_id="m1", batter="P1", runs_batter=np.int64(3))])
    out = tmp_path / "agg"
    res = aggregate_all(str(root), str(out), force=True)
    assert len(res["career_batting"]) == 1
    assert res["career_batting"].iloc[0]["player"] == "P1"


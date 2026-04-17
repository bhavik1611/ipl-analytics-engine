from __future__ import annotations

import numpy as np
import pandas as pd

from src.scoring.calculator import calculate_match_points


def make_delivery(**kwargs: object) -> dict[str, object]:
    row: dict[str, object] = {
        "match_id": "m1",
        "season": np.int64(2024),
        "date": "2024-01-01",
        "venue": "Test Venue",
        "city": "Test City",
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
        "phase": "pp",
        "is_legal_delivery": True,
        "batter": "BAT",
        "non_striker": "NS",
        "runs_batter": np.int64(0),
        "is_four": False,
        "is_six": False,
        "is_dot_ball": True,
        "bowler": "BOWL",
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


def _get_player_points(df: pd.DataFrame, player: str) -> tuple[int, int, int, int]:
    out = calculate_match_points(df)
    r = out.loc[out["player"] == player].iloc[0]
    return (
        int(r["fantasy_batting"]),
        int(r["fantasy_bowling"]),
        int(r["fantasy_fielding"]),
        int(r["fantasy_total"]),
    )


def test_batting_basic_runs_not_out() -> None:
    rows = [make_delivery(batter="P1", runs_batter=np.int64(1)) for _ in range(35)]
    df = pd.DataFrame(rows)
    b, _, _, t = _get_player_points(df, "P1")
    assert b == 45
    assert t == 45


def test_batting_duck_dismissed() -> None:
    df = pd.DataFrame(
        [
            make_delivery(batter="P1", runs_batter=np.int64(0)),
            make_delivery(
                batter="P1",
                is_wicket=True,
                wicket_player_out="P1",
                wicket_kind="caught",
                is_bowler_wicket=True,
            ),
        ]
    )
    b, _, _, t = _get_player_points(df, "P1")
    assert b == -10
    assert t == -10


def test_batting_duck_plus_run_out() -> None:
    df = pd.DataFrame(
        [
            make_delivery(batter="P1", runs_batter=np.int64(0)),
            make_delivery(batter="P1", is_wicket=True, wicket_player_out="P1", wicket_kind="run out"),
        ]
    )
    b, _, _, t = _get_player_points(df, "P1")
    assert b == -20
    assert t == -20


def test_batting_retired_hurt_40() -> None:
    rows = [make_delivery(batter="P1", runs_batter=np.int64(1)) for _ in range(40)]
    rows.append(make_delivery(batter="P1", is_wicket=True, wicket_player_out="P1", wicket_kind="retired hurt"))
    df = pd.DataFrame(rows)
    b, _, _, _ = _get_player_points(df, "P1")
    assert b == 40


def test_batting_retired_out_30() -> None:
    rows = [make_delivery(batter="P1", runs_batter=np.int64(1)) for _ in range(30)]
    rows.append(make_delivery(batter="P1", is_wicket=True, wicket_player_out="P1", wicket_kind="retired out"))
    df = pd.DataFrame(rows)
    b, _, _, _ = _get_player_points(df, "P1")
    assert b == 20


def test_batting_50_milestone_not_out() -> None:
    rows = [make_delivery(batter="P1", runs_batter=np.int64(1)) for _ in range(50)]
    df = pd.DataFrame(rows)
    b, _, _, _ = _get_player_points(df, "P1")
    assert b == 70


def test_batting_75_milestone_not_out() -> None:
    rows = [make_delivery(batter="P1", runs_batter=np.int64(1)) for _ in range(75)]
    df = pd.DataFrame(rows)
    b, _, _, _ = _get_player_points(df, "P1")
    assert b == 105


def test_batting_100_milestone_dismissed() -> None:
    rows = [make_delivery(batter="P1", runs_batter=np.int64(1)) for _ in range(100)]
    rows.append(
        make_delivery(batter="P1", is_wicket=True, wicket_player_out="P1", wicket_kind="caught", is_bowler_wicket=True)
    )
    df = pd.DataFrame(rows)
    b, _, _, _ = _get_player_points(df, "P1")
    assert b == 130


def test_batting_not_out_zero_runs_balls_faced() -> None:
    df = pd.DataFrame(
        [
            make_delivery(batter="P1", runs_batter=np.int64(0)),
            make_delivery(batter="P1", runs_batter=np.int64(0)),
        ]
    )
    b, _, _, _ = _get_player_points(df, "P1")
    assert b == 10


def test_bowling_basic_wickets_2_caught() -> None:
    df = pd.DataFrame(
        [
            make_delivery(
                bowler="P2",
                runs_batter=np.int64(1),
                runs_bowler=np.int64(1),
                is_wicket=True,
                is_bowler_wicket=True,
                wicket_kind="caught",
                wicket_player_out="X",
            ),
            make_delivery(
                bowler="P2",
                runs_batter=np.int64(1),
                runs_bowler=np.int64(1),
                is_wicket=True,
                is_bowler_wicket=True,
                wicket_kind="caught",
                wicket_player_out="Y",
            ),
        ]
    )
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == 40


def test_bowling_bowled_plus_lbw() -> None:
    df = pd.DataFrame(
        [
            make_delivery(
                bowler="P2",
                runs_batter=np.int64(1),
                runs_bowler=np.int64(1),
                is_wicket=True,
                is_bowler_wicket=True,
                wicket_kind="bowled",
                wicket_player_out="X",
            ),
            make_delivery(
                bowler="P2",
                runs_batter=np.int64(1),
                runs_bowler=np.int64(1),
                is_wicket=True,
                is_bowler_wicket=True,
                wicket_kind="lbw",
                wicket_player_out="Y",
            ),
        ]
    )
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == 60


def test_bowling_maiden_over() -> None:
    rows = [
        make_delivery(bowler="P2", over_1indexed=np.int64(1), ball=np.int64(i + 1), runs_batter=np.int64(0), runs_bowler=np.int64(0))
        for i in range(6)
    ]
    df = pd.DataFrame(rows)
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == 30


def test_bowling_maiden_broken_by_wide() -> None:
    rows = [
        make_delivery(bowler="P2", over_1indexed=np.int64(1), ball=np.int64(1), runs_batter=np.int64(0), runs_bowler=np.int64(0))
    ]
    rows.append(
        make_delivery(
            bowler="P2",
            over_1indexed=np.int64(1),
            ball=np.int64(2),
            is_wide=True,
            is_legal_delivery=False,
            wide_runs=np.int64(1),
            extras_total=np.int64(1),
            runs_bowler=np.int64(1),
            runs_batter=np.int64(0),
        )
    )
    rows.extend(
        [
            make_delivery(bowler="P2", over_1indexed=np.int64(1), ball=np.int64(i + 3), runs_batter=np.int64(0), runs_bowler=np.int64(0))
            for i in range(4)
        ]
    )
    df = pd.DataFrame(rows)
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == 0


def test_bowling_maiden_broken_by_noball() -> None:
    rows = [
        make_delivery(bowler="P2", over_1indexed=np.int64(1), ball=np.int64(1), runs_batter=np.int64(0), runs_bowler=np.int64(0))
    ]
    rows.append(
        make_delivery(
            bowler="P2",
            over_1indexed=np.int64(1),
            ball=np.int64(2),
            is_noball=True,
            is_legal_delivery=False,
            noball_runs=np.int64(1),
            extras_total=np.int64(1),
            runs_bowler=np.int64(1),
            runs_batter=np.int64(0),
        )
    )
    rows.extend(
        [
            make_delivery(bowler="P2", over_1indexed=np.int64(1), ball=np.int64(i + 3), runs_batter=np.int64(0), runs_bowler=np.int64(0))
            for i in range(4)
        ]
    )
    df = pd.DataFrame(rows)
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == 0


def test_bowling_4_wicket_haul_bonus() -> None:
    rows = [
        make_delivery(
            bowler="P2",
            runs_batter=np.int64(1),
            runs_bowler=np.int64(1),
            is_wicket=True,
            is_bowler_wicket=True,
            wicket_kind="caught",
            wicket_player_out=f"X{i}",
        )
        for i in range(4)
    ]
    df = pd.DataFrame(rows)
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == 100


def test_bowling_5_wicket_haul_bonus_once() -> None:
    rows = [
        make_delivery(
            bowler="P2",
            runs_batter=np.int64(1),
            runs_bowler=np.int64(1),
            is_wicket=True,
            is_bowler_wicket=True,
            wicket_kind="caught",
            wicket_player_out=f"X{i}",
        )
        for i in range(5)
    ]
    df = pd.DataFrame(rows)
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == 120


def test_bowling_expensive_50() -> None:
    df = pd.DataFrame([make_delivery(bowler="P2", runs_bowler=np.int64(50))])
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == -10


def test_bowling_expensive_60() -> None:
    df = pd.DataFrame([make_delivery(bowler="P2", runs_bowler=np.int64(60))])
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == -20


def test_bowling_expensive_70() -> None:
    df = pd.DataFrame([make_delivery(bowler="P2", runs_bowler=np.int64(70))])
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == -30


def test_bowling_economy_bonus() -> None:
    rows = [
        make_delivery(
            bowler="P2",
            over_1indexed=np.int64(o),
            ball=np.int64(b),
            is_legal_delivery=True,
            runs_batter=np.int64(0),
            runs_bowler=np.int64(0),
        )
        for o in range(1, 5)
        for b in range(1, 7)
    ]
    for i in range(22):
        rows[i]["runs_batter"] = np.int64(1)
        rows[i]["runs_bowler"] = np.int64(1)
    df = pd.DataFrame(rows)
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == 10


def test_bowling_economy_boundary_exact_6() -> None:
    rows = [
        make_delivery(
            bowler="P2",
            over_1indexed=np.int64(o),
            ball=np.int64(b),
            is_legal_delivery=True,
            runs_batter=np.int64(1),
            runs_bowler=np.int64(1),
        )
        for o in range(1, 5)
        for b in range(1, 7)
    ]
    df = pd.DataFrame(rows)
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == 0


def test_bowling_economy_insufficient_overs() -> None:
    rows = [
        make_delivery(
            bowler="P2",
            over_1indexed=np.int64(o),
            ball=np.int64(b),
            is_legal_delivery=True,
            runs_batter=np.int64(0),
            runs_bowler=np.int64(0),
        )
        for o in range(1, 3)
        for b in range(1, 7)
    ]
    for i in range(8):
        rows[i]["runs_batter"] = np.int64(1)
        rows[i]["runs_bowler"] = np.int64(1)
    df = pd.DataFrame(rows)
    _, bo, _, _ = _get_player_points(df, "P2")
    assert bo == 0


def test_fielding_catch() -> None:
    df = pd.DataFrame(
        [
            make_delivery(is_wicket=True, wicket_kind="caught", wicket_fielder1="F1", fielder1_is_sub=False),
        ]
    )
    _, _, f, _ = _get_player_points(df, "F1")
    assert f == 10


def test_fielding_caught_and_bowled() -> None:
    df = pd.DataFrame(
        [
            make_delivery(is_wicket=True, wicket_kind="caught and bowled", wicket_fielder1="P2", bowler="P2", fielder1_is_sub=None),
        ]
    )
    _, _, f, _ = _get_player_points(df, "P2")
    assert f == 10


def test_fielding_caught_and_bowled_missing_fielder_credits_bowler() -> None:
    df = pd.DataFrame(
        [
            make_delivery(
                is_wicket=True,
                wicket_kind="caught and bowled",
                wicket_fielder1=np.nan,
                bowler="P2",
            ),
        ]
    )
    _, _, f, _ = _get_player_points(df, "P2")
    assert f == 10


def test_fielding_run_out_non_sub() -> None:
    df = pd.DataFrame(
        [
            make_delivery(is_wicket=True, wicket_kind="run out", wicket_fielder1="F1", fielder1_is_sub=False),
        ]
    )
    _, _, f, _ = _get_player_points(df, "F1")
    assert f == 20


def test_fielding_run_out_two_fielders_split() -> None:
    df = pd.DataFrame(
        [
            make_delivery(
                is_wicket=True,
                wicket_kind="run out",
                wicket_fielder1="F1",
                wicket_fielder2="F2",
                fielder1_is_sub=False,
                fielder2_is_sub=False,
            ),
        ]
    )
    _, _, f1, _ = _get_player_points(df, "F1")
    _, _, f2, _ = _get_player_points(df, "F2")
    assert f1 == 10
    assert f2 == 10


def test_fielding_run_out_substitute() -> None:
    df = pd.DataFrame(
        [
            make_delivery(is_wicket=True, wicket_kind="run out", wicket_fielder1="F1", fielder1_is_sub=True),
        ]
    )
    _, _, f, _ = _get_player_points(df, "F1")
    assert f == 0


def test_fielding_stumping() -> None:
    df = pd.DataFrame(
        [
            make_delivery(is_wicket=True, wicket_kind="stumped", wicket_fielder1="WK", fielder1_is_sub=False),
        ]
    )
    _, _, f, _ = _get_player_points(df, "WK")
    assert f == 10


def test_fielding_fielder1_is_sub_none_treated_false() -> None:
    df = pd.DataFrame(
        [
            make_delivery(is_wicket=True, wicket_kind="run out", wicket_fielder1="F1", fielder1_is_sub=None),
        ]
    )
    _, _, f, _ = _get_player_points(df, "F1")
    assert f == 20


def test_super_over_exclusion() -> None:
    df = pd.DataFrame(
        [
            make_delivery(batter="P1", runs_batter=np.int64(10), is_super_over=False),
            make_delivery(batter="P1", runs_batter=np.int64(10), is_super_over=True),
        ]
    )
    b, _, _, _ = _get_player_points(df, "P1")
    assert b == 20


def test_combined_match_batting_plus_fielding() -> None:
    rows = [make_delivery(batter="P1", runs_batter=np.int64(1)) for _ in range(30)]
    rows.append(make_delivery(is_wicket=True, wicket_kind="caught", wicket_fielder1="P1", fielder1_is_sub=False))
    rows.append(make_delivery(is_wicket=True, wicket_kind="caught", wicket_fielder1="P1", fielder1_is_sub=False))
    df = pd.DataFrame(rows)
    b, _, f, t = _get_player_points(df, "P1")
    assert b == 40
    assert f == 20
    assert t == 60


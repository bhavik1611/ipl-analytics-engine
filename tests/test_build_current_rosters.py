from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.scripts.build_current_rosters as bcr


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_max_season_and_pairs_latest_season_only(tmp_path: Path) -> None:
    matches = tmp_path / "matches"
    _write_parquet(
        matches / "a.parquet",
        [
            {
                "season": np.int64(2024),
                "batter": "Old Only",
                "bowler": "P2",
                "batting_team": "A",
                "bowling_team": "B",
            }
        ],
    )
    _write_parquet(
        matches / "b.parquet",
        [
            {
                "season": np.int64(2025),
                "batter": "Active",
                "bowler": "BowlerX",
                "batting_team": "A",
                "bowling_team": "B",
            }
        ],
    )
    paths = bcr._list_parquet_paths(matches)
    peak = bcr._max_season_from_parquets(paths)
    assert peak == 2025
    pairs = bcr._pairs_for_season(paths, peak)
    assert ("A", "Active") in pairs
    assert ("B", "BowlerX") in pairs
    assert ("A", "Old Only") not in pairs


def test_write_outputs_empty_and_nonempty(tmp_path: Path) -> None:
    csv_p = tmp_path / "r.csv"
    json_p = tmp_path / "r.json"
    bcr._write_outputs(csv_p, json_p, set(), 0)
    df = pd.read_csv(csv_p)
    assert df.empty
    assert list(df.columns) == bcr._CSV_COLUMNS

    bcr._write_outputs(csv_p, json_p, {("T", "P")}, 2025)
    df2 = pd.read_csv(csv_p)
    assert len(df2) == 1
    assert int(df2.iloc[0]["season"]) == 2025
    assert df2.iloc[0]["team"] == "T"
    assert df2.iloc[0]["player"] == "P"
    role_cell = df2.iloc[0]["role"]
    assert pd.isna(role_cell) or str(role_cell).strip() == ""
    assert bool(df2.iloc[0]["is_captain"]) is False


def test_merge_roles_from_existing_json(tmp_path: Path) -> None:
    """Hand-edited JSON overrides role and captain before parquet-driven write."""

    json_p = tmp_path / "r.json"
    doc = {
        "season": 2025,
        "as_of_date": "2025-01-01",
        "source": bcr._SOURCE,
        "teams": {
            "T": [{"name": "P", "role": ["Batsman", "Spin Bowler"], "is_captain": True}]
        },
    }
    json_p.write_text(json.dumps(doc), encoding="utf-8")
    csv_p = tmp_path / "r.csv"
    bcr._write_outputs(csv_p, json_p, {("T", "P")}, 2025)
    df = pd.read_csv(csv_p)
    assert df.iloc[0]["role"] == "Batsman; Spin Bowler"
    assert bool(df.iloc[0]["is_captain"]) is True


def test_roster_csv_rows_from_json_document() -> None:
    """Flatten JSON with list roles to CSV row dicts."""

    rows = bcr.roster_csv_rows_from_json_document(
        {
            "season": 2026,
            "source": "manual",
            "as_of_date": "2026-04-01",
            "teams": {"A": [{"name": "X", "role": ["Pace Bowler"], "is_captain": False}]},
        }
    )
    assert len(rows) == 1
    assert rows[0]["team"] == "A"
    assert rows[0]["player"] == "X"
    assert rows[0]["role"] == "Pace Bowler"
    assert rows[0]["season"] == 2026
    assert rows[0]["source"] == "manual"

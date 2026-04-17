import logging
import shutil
from pathlib import Path

import pandas as pd

from src.utils.parser import _COLUMNS, _backfill_innings_totals, parse_all_matches


def _run_parse(tmp_path: Path, fixture_name: str, caplog) -> pd.DataFrame:
    src = Path(__file__).parent / "fixtures" / fixture_name
    dst = tmp_path / fixture_name
    shutil.copyfile(src, dst)
    caplog.set_level(logging.DEBUG)
    processed_dir = tmp_path / "processed"
    return parse_all_matches(str(tmp_path), str(processed_dir))


def test_basic_delivery_parsing(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 1)].iloc[0]
    assert row["runs_batter"] == 1
    assert row["batter"] == "A1"
    assert row["bowler"] == "B1"


def test_wide_delivery(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 2)].iloc[0]
    assert bool(row["is_wide"]) is True
    assert bool(row["is_legal_delivery"]) is False
    assert row["wide_runs"] == 1
    assert row["runs_bowler"] == 1


def test_noball_delivery(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 3)].iloc[0]
    assert bool(row["is_noball"]) is True
    assert bool(row["is_legal_delivery"]) is False
    assert row["noball_runs"] == 1
    assert row["runs_bowler"] == 5


def test_bye_delivery(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 4)].iloc[0]
    assert row["bye_runs"] == 2
    assert row["runs_bowler"] == 0


def test_legbye_delivery(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 5)].iloc[0]
    assert row["legbye_runs"] == 1
    assert row["runs_bowler"] == 0


def test_wicket_bowled(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 6)].iloc[0]
    assert bool(row["is_wicket"]) is True
    assert bool(row["is_bowler_wicket"]) is True
    assert row["wicket_kind"] == "bowled"
    assert pd.isna(row["wicket_fielder1"]) or row["wicket_fielder1"] is None


def test_wicket_caught_fielder(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 7)].iloc[0]
    assert row["wicket_kind"] == "caught"
    assert row["wicket_fielder1"] == "F1"
    assert bool(row["fielder1_is_sub"]) is False


def test_run_out_substitute_fielder(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 8)].iloc[0]
    assert row["wicket_kind"] == "run out"
    assert row["wicket_fielder1"] == "Sub Fielder"
    assert bool(row["fielder1_is_sub"]) is True
    assert bool(row["is_bowler_wicket"]) is False


def test_super_over_detection(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 3) & (df["over"] == 20) & (df["ball"] == 1)].iloc[0]
    assert bool(row["is_super_over"]) is True
    assert row["phase"] == "super_over"


def test_two_wickets_on_one_delivery(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 9)].iloc[0]
    assert row["wicket_player_out"] == "A4"
    assert row["wicket2_player_out"] == "A1"
    assert row["wicket2_kind"] == "caught"
    assert row["wicket2_fielder1"] == "F3"


def test_is_playoff_true_final(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    assert bool(df["is_playoff"].iloc[0]) is True


def test_is_playoff_false_stage_absent(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_no_stage.json", caplog)
    assert bool(df["is_playoff"].iloc[0]) is False


def test_is_playoff_warning_unrecognised(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_bad_stage.json", caplog)
    assert bool(df["is_playoff"].iloc[0]) is False
    assert any("Unrecognised event.stage value" in rec.message for rec in caplog.records)


def test_backfill_innings_total_wickets_and_target(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    first_innings = df[df["innings_num"] == 1]
    assert int(first_innings["innings_total"].iloc[0]) == 10
    assert int(first_innings["innings_wickets"].iloc[0]) == 4
    second_innings = df[df["innings_num"] == 2]
    assert int(second_innings["target"].iloc[0]) == 11


def test_season_fallback_debug_log(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_no_season.json", caplog)
    assert int(df["season"].iloc[0]) == 2019
    assert any("Season missing; derived from date" in rec.message for rec in caplog.records)


def test_column_completeness(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    assert list(df.columns) == _COLUMNS
    assert len(set(df.columns)) == len(_COLUMNS)


def test_registry_populates_batter_bowler_ids(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 1)].iloc[0]
    assert row["batter_id"] == "a0000001"
    assert row["non_striker_id"] == "a0000002"
    assert row["bowler_id"] == "b0000001"


def test_registry_populates_wicket_and_fielder_ids(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    caught = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 7)].iloc[0]
    assert caught["wicket_player_out_id"] == "a0000002"
    assert caught["wicket_fielder1_id"] == "f0000001"
    dual = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 9)].iloc[0]
    assert dual["wicket2_player_out_id"] == "a0000001"
    assert dual["wicket2_fielder1_id"] == "f0000003"


def test_fielder_not_in_registry_has_null_id(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_all_cases_final.json", caplog)
    row = df[(df["innings_num"] == 1) & (df["over"] == 0) & (df["ball"] == 8)].iloc[0]
    assert row["wicket_fielder1"] == "Sub Fielder"
    assert pd.isna(row["wicket_fielder1_id"]) or row["wicket_fielder1_id"] is None


def test_no_registry_id_columns_nullable(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_no_season.json", caplog)
    row = df.iloc[0]
    assert pd.isna(row["batter_id"]) or row["batter_id"] is None
    assert list(df.columns) == _COLUMNS


def test_officials_in_registry_do_not_break_parsing(tmp_path: Path, caplog) -> None:
    df = _run_parse(tmp_path, "match_registry_officials_deliveries.json", caplog)
    assert len(df) == 1
    assert df.iloc[0]["batter_id"] == "a0000001"


def test_backfill_empty_df() -> None:
    df = pd.DataFrame([])
    out = _backfill_innings_totals(df)
    assert list(out.columns) == _COLUMNS


def test_incremental_skip_existing_parquet(tmp_path: Path, caplog) -> None:
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fixture = Path(__file__).parent / "fixtures" / "match_all_cases_final.json"
    target = raw_dir / "12345.json"
    shutil.copyfile(fixture, target)

    caplog.set_level(logging.DEBUG)
    df1 = parse_all_matches(str(raw_dir), str(processed_dir), force=False)
    assert len(df1) > 0

    target.write_text("{ not json", encoding="utf-8")
    caplog.clear()
    df2 = parse_all_matches(str(raw_dir), str(processed_dir), force=False)
    assert any("Skipping 12345 — already processed" in rec.message for rec in caplog.records)
    assert len(df2) == len(df1)


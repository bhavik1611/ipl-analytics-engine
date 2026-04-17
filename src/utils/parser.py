"""Cricsheet IPL JSON parser.

This module flattens Cricsheet-style match JSON into a delivery-level pandas
DataFrame used by later pipeline phases.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_project_paths, load_env

logger = logging.getLogger(__name__)


PLAYOFF_STAGES: set[str] = {
    "Qualifier 1", "Qualifier 2", "Eliminator", "Final",
    "Elimination Final", "Semi Final", "3rd Place Play-Off"
}

_BOWLER_WICKET_KINDS: set[str] = {
    "bowled",
    "lbw",
    "caught",
    "caught and bowled",
    "stumped",
    "hit wicket",
}

_EXTRAS_KEY_MAP: dict[str, str] = {
    "wides": "wide",
    "noballs": "noball",
    "byes": "bye",
    "legbyes": "legbye",
    "penalty": "penalty",
    "wide": "wide",
    "noball": "noball",
    "bye": "bye",
    "legbye": "legbye",
}

_COLUMNS: list[str] = [
    # MATCH CONTEXT
    "match_id",
    "season",
    "date",
    "venue",
    "city",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "match_winner",
    "win_by_runs",
    "win_by_wickets",
    "player_of_match",
    "is_playoff",
    "match_number",
    # INNINGS CONTEXT
    "innings_num",
    "is_super_over",
    "batting_team",
    "bowling_team",
    "innings_total",
    "innings_wickets",
    "target",
    # DELIVERY CONTEXT
    "over",
    "over_1indexed",
    "ball",
    "phase",
    "is_legal_delivery",
    # BATTING
    "batter",
    "non_striker",
    "runs_batter",
    "is_four",
    "is_six",
    "is_dot_ball",
    # BOWLING
    "bowler",
    "runs_bowler",
    "is_wide",
    "is_noball",
    "wide_runs",
    "noball_runs",
    "bye_runs",
    "legbye_runs",
    # EXTRAS
    "extras_total",
    "extras_type",
    # WICKET
    "is_wicket",
    "wicket_player_out",
    "wicket_kind",
    "wicket_fielder1",
    "wicket_fielder2",
    "fielder1_is_sub",
    "fielder2_is_sub",
    "is_bowler_wicket",
    # SECOND WICKET
    "wicket2_player_out",
    "wicket2_kind",
    "wicket2_fielder1",
    "wicket2_fielder2",
    "wicket2_fielder1_is_sub",
    "wicket2_fielder2_is_sub",
    # Cricsheet registry.people ids (nullable; absent registry → null/NaN)
    "batter_id",
    "non_striker_id",
    "bowler_id",
    "wicket_player_out_id",
    "wicket_fielder1_id",
    "wicket_fielder2_id",
    "wicket2_player_out_id",
    "wicket2_fielder1_id",
    "wicket2_fielder2_id",
]

# Subset of _COLUMNS used to backfill older parquet files in downstream phases.
PEOPLE_ID_DELIVERY_COLUMNS: tuple[str, ...] = (
    "batter_id",
    "non_striker_id",
    "bowler_id",
    "wicket_player_out_id",
    "wicket_fielder1_id",
    "wicket_fielder2_id",
    "wicket2_player_out_id",
    "wicket2_fielder1_id",
    "wicket2_fielder2_id",
)


def parse_all_matches(raw_dir: str, processed_dir: str, force: bool = False) -> pd.DataFrame:
    """Parse all Cricsheet JSON matches under a directory.

    Args:
        raw_dir: Directory containing `*.json` match files. If empty, uses env
            CRICSHEET_RAW_DIR defaulting to `./data/raw/cricsheet_ipl`.
        processed_dir: Directory for processed outputs; if empty, uses env
            PROCESSED_DIR defaulting to `./data/processed`.
        force: If True, reparse and overwrite all match parquet files.

    Returns:
        Delivery-level DataFrame with the exact output contract columns.
    """
    load_env()
    paths = get_project_paths()
    resolved_raw_dir = raw_dir.strip() if raw_dir is not None else ""
    resolved_processed_dir = processed_dir.strip() if processed_dir is not None else ""

    start = time.time()
    raw_path = Path(resolved_raw_dir) if resolved_raw_dir else paths.cricsheet_raw_dir
    filepaths = sorted(raw_path.glob("*.json"))
    stage_values: set[str] = set()
    out_root = Path(resolved_processed_dir) if resolved_processed_dir else paths.processed_dir
    matches_dir = out_root / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    for idx, filepath in enumerate(filepaths, start=1):
        match_id = filepath.stem
        out_path = matches_dir / f"{match_id}.parquet"
        if out_path.exists() and not force:
            logger.debug("Skipping %s — already processed", match_id)
            continue

        rows, stage = _parse_match(filepath)
        if stage is not None:
            stage_values.add(stage)
        _write_match_parquet(match_id=match_id, rows=rows, out_path=out_path)
        if idx % 100 == 0:
            logger.info(
                "Parsed %s/%s files in %.1fs",
                idx,
                len(filepaths),
                time.time() - start,
            )

    df = _read_all_match_parquets(matches_dir)

    logger.info("Total deliveries parsed: %s", len(df))
    logger.info("Total matches parsed: %s", len(filepaths))
    logger.info("Super over deliveries: %s", int(df["is_super_over"].fillna(False).sum()))
    logger.info("Distinct event.stage values: %s", sorted(stage_values))

    return df


def _write_match_parquet(match_id: str, rows: list[dict[str, Any]], out_path: Path) -> None:
    """Write one match parquet file, never raising.

    Args:
        match_id: Match id used for logging.
        rows: Delivery rows for the match.
        out_path: Full parquet file output path.
    """
    try:
        df = pd.DataFrame(rows)
        df = _backfill_innings_totals(df)
        df = df.reindex(columns=_COLUMNS)
        df.to_parquet(out_path, index=False)
    except (OSError, ValueError, ImportError) as exc:
        logger.error("Failed writing parquet for %s: %s", match_id, exc)


def _read_all_match_parquets(matches_dir: Path) -> pd.DataFrame:
    """Read all match parquet files into one DataFrame.

    Args:
        matches_dir: Directory containing match parquet files.

    Returns:
        Combined DataFrame with contract columns.
    """
    filepaths = sorted(matches_dir.glob("*.parquet"))
    if not filepaths:
        return pd.DataFrame([]).reindex(columns=_COLUMNS)
    frames: list[pd.DataFrame] = []
    for fp in filepaths:
        frames.append(pd.read_parquet(fp))
    out = pd.concat(frames, ignore_index=True)
    return out.reindex(columns=_COLUMNS)


def _parse_match(filepath: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Parse a single match file into delivery rows.

    Args:
        filepath: Path to a Cricsheet JSON match file.

    Returns:
        Tuple of (rows, stage_value_if_present).
    """
    try:
        payload = json.loads(filepath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed reading match JSON %s: %s", filepath, exc)
        return ([], None)

    info = payload.get("info", {})
    match_id = filepath.stem

    dates = info.get("dates", [])
    date = _parse_date(dates[0]) if dates else None
    season = _derive_season(info)

    teams = info.get("teams", [])
    team1 = teams[0] if len(teams) > 0 else None
    team2 = teams[1] if len(teams) > 1 else None

    toss = info.get("toss", {})
    outcome = info.get("outcome", {})
    outcome_by = outcome.get("by", {}) if isinstance(outcome.get("by", {}), dict) else {}

    pom = info.get("player_of_match", [])
    if isinstance(pom, list):
        player_of_match = ", ".join([str(p) for p in pom])
    else:
        player_of_match = str(pom) if pom is not None else None

    event = info.get("event", {}) if isinstance(info.get("event", {}), dict) else {}
    stage = event.get("stage")

    match_meta: dict[str, Any] = {
        "match_id": match_id,
        "season": season,
        "date": date,
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team1": team1,
        "team2": team2,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "match_winner": outcome.get("winner"),
        "win_by_runs": outcome_by.get("runs"),
        "win_by_wickets": outcome_by.get("wickets"),
        "player_of_match": player_of_match,
        "is_playoff": _derive_is_playoff(info),
        "match_number": event.get("match_number"),
    }

    people_map = _extract_people_registry(info)
    innings_list = payload.get("innings", [])
    rows: list[dict[str, Any]] = []
    for innings_idx, innings in enumerate(innings_list, start=1):
        rows.extend(_parse_innings(match_meta, innings, innings_idx, people_map))

    return (rows, stage if isinstance(stage, str) else None)


def _parse_innings(
    match_meta: dict[str, Any],
    innings: dict[str, Any],
    innings_num: int,
    people_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Parse an innings into delivery rows.

    Args:
        match_meta: Match-level columns to include per row.
        innings: Innings dictionary from Cricsheet.
        innings_num: 1-based innings number.
        people_map: Exact display name → Cricsheet people id from ``info.registry.people``.

    Returns:
        List of delivery rows as dictionaries.
    """
    batting_team = innings.get("team")
    team1 = match_meta.get("team1")
    team2 = match_meta.get("team2")
    bowling_team = team2 if batting_team == team1 else team1 if batting_team == team2 else None

    base_row: dict[str, Any] = {
        **match_meta,
        "innings_num": innings_num,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "innings_total": None,
        "innings_wickets": None,
        "target": None,
    }

    overs = innings.get("overs", [])
    rows: list[dict[str, Any]] = []
    for over_obj in overs:
        over_idx = over_obj.get("over")
        deliveries = over_obj.get("deliveries", [])
        if over_idx is None:
            continue
        for ball_idx, delivery in enumerate(deliveries, start=1):
            rows.append(
                _parse_delivery(base_row, int(over_idx), ball_idx, delivery, people_map)
            )
    return rows


def _parse_delivery(
    base_row: dict[str, Any],
    over_idx: int,
    ball_idx: int,
    delivery: dict[str, Any],
    people_map: dict[str, str],
) -> dict[str, Any]:
    """Parse one delivery.

    Args:
        base_row: Base row fields (match+innings).
        over_idx: 0-indexed over number from JSON.
        ball_idx: 1-indexed ball number within the over.
        delivery: Delivery dictionary from JSON.
        people_map: Name → people id for this file (may be empty).

    Returns:
        Row dictionary for a single delivery.
    """
    runs = delivery.get("runs", {})
    runs_batter = int(runs.get("batter", 0) or 0)
    extras_total = int(runs.get("extras", 0) or 0)

    extras_dict = delivery.get("extras", {})
    extras_key = None
    extras_runs_value = 0
    if isinstance(extras_dict, dict) and extras_dict:
        keys = list(extras_dict.keys())
        if len(keys) > 1:
            logger.warning("Multiple extras keys on one delivery; using first: %s", keys)
        extras_key = keys[0]
        extras_runs_value = int(extras_dict.get(extras_key, 0) or 0)

    extras_type = _EXTRAS_KEY_MAP.get(str(extras_key), None) if extras_key else None
    is_wide = extras_type == "wide"
    is_noball = extras_type == "noball"

    wide_runs = extras_runs_value if is_wide else 0
    noball_runs = extras_runs_value if is_noball else 0
    bye_runs = extras_runs_value if extras_type == "bye" else 0
    legbye_runs = extras_runs_value if extras_type == "legbye" else 0

    is_super_over = over_idx == 20
    phase = _derive_phase(over_idx, is_super_over)
    is_legal_delivery = not (is_wide or is_noball)

    wickets = delivery.get("wickets", [])
    wicket1 = wickets[0] if isinstance(wickets, list) and len(wickets) >= 1 else None
    wicket2 = wickets[1] if isinstance(wickets, list) and len(wickets) >= 2 else None

    wicket1_parsed = _parse_wicket(wicket1, people_map)
    wicket2_parsed = _parse_wicket(wicket2, people_map)

    is_wicket = wicket1 is not None
    wicket_kind = wicket1_parsed.get("wicket_kind")
    is_bowler_wicket = bool(wicket_kind in _BOWLER_WICKET_KINDS) if wicket_kind else False

    row: dict[str, Any] = {
        **base_row,
        "is_super_over": is_super_over,
        "over": over_idx,
        "over_1indexed": over_idx + 1,
        "ball": ball_idx,
        "phase": phase,
        "is_legal_delivery": is_legal_delivery,
        "batter": delivery.get("batter"),
        "non_striker": delivery.get("non_striker"),
        "runs_batter": runs_batter,
        "is_four": runs_batter == 4,
        "is_six": runs_batter == 6,
        "is_dot_ball": runs_batter == 0 and is_legal_delivery,
        "bowler": delivery.get("bowler"),
        "runs_bowler": runs_batter + wide_runs + noball_runs,
        "is_wide": is_wide,
        "is_noball": is_noball,
        "wide_runs": wide_runs,
        "noball_runs": noball_runs,
        "bye_runs": bye_runs,
        "legbye_runs": legbye_runs,
        "extras_total": extras_total,
        "extras_type": extras_type,
        "is_wicket": is_wicket,
        **wicket1_parsed,
        "is_bowler_wicket": is_bowler_wicket,
        **_prefix_keys(wicket2_parsed, "wicket2_"),
        "batter_id": _people_id(people_map, delivery.get("batter")),
        "non_striker_id": _people_id(people_map, delivery.get("non_striker")),
        "bowler_id": _people_id(people_map, delivery.get("bowler")),
    }

    return row


def _backfill_innings_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill innings_total, innings_wickets and target.

    Args:
        df: Parsed delivery DataFrame (may be empty).

    Returns:
        DataFrame with innings totals and chase targets populated.
    """
    if df.empty:
        return df.reindex(columns=_COLUMNS)

    total_runs = (df["runs_batter"].fillna(0) + df["extras_total"].fillna(0)).astype("int64")
    totals = (
        df.assign(_total_runs=total_runs)
        .groupby(["match_id", "innings_num"], dropna=False)["_total_runs"]
        .sum()
        .rename("innings_total")
    )
    wickets = (
        df["is_wicket"]
        .fillna(False)
        .groupby([df["match_id"], df["innings_num"]], dropna=False)
        .sum()
        .astype("int64")
        .rename("innings_wickets")
    )

    out = df.join(totals, on=["match_id", "innings_num"], rsuffix="_bf")
    out = out.join(wickets, on=["match_id", "innings_num"], rsuffix="_bf")
    if "innings_total_bf" in out.columns:
        out["innings_total"] = out["innings_total_bf"]
        out = out.drop(columns=["innings_total_bf"])
    if "innings_wickets_bf" in out.columns:
        out["innings_wickets"] = out["innings_wickets_bf"]
        out = out.drop(columns=["innings_wickets_bf"])

    innings1_totals = totals.reset_index()
    innings1_totals = innings1_totals[innings1_totals["innings_num"] == 1].set_index("match_id")[
        "innings_total"
    ]
    out["target"] = pd.NA
    mask = (out["innings_num"] == 2) & (~out["is_super_over"].fillna(False))
    out.loc[mask, "target"] = out.loc[mask, "match_id"].map(innings1_totals).astype("Int64") + 1

    return out


def _derive_season(info: dict[str, Any]) -> int | None:
    """Derive season as an integer year.

    Args:
        info: The `info` object from Cricsheet JSON.

    Returns:
        Season year (int) or None if not derivable.
    """
    season_raw = info.get("season")
    if season_raw is not None:
        digits = "".join([c for c in str(season_raw) if c.isdigit()])
        if len(digits) >= 4:
            return int(digits[:4])

    dates = info.get("dates", [])
    if dates:
        date = _parse_date(dates[0])
        if date is not None:
            logger.debug("Season missing; derived from date %s", date.isoformat())
            return int(date.year)
    logger.debug("Season missing and date unavailable; returning None")
    return None


def _derive_is_playoff(info: dict[str, Any]) -> bool:
    """Derive playoff flag from `info.event.stage`.

    Args:
        info: The `info` object from Cricsheet JSON.

    Returns:
        True for known playoff stages, False otherwise.
    """
    event = info.get("event", {})
    if not isinstance(event, dict):
        return False
    stage = event.get("stage")
    if stage is None:
        return False
    if stage in PLAYOFF_STAGES:
        return True
    logger.warning("Unrecognised event.stage value: %r", stage)
    return False


def _parse_date(value: Any) -> dt.date | None:
    """Parse a Cricsheet date field into a datetime.date.

    Args:
        value: Date value from JSON.

    Returns:
        Parsed date or None.
    """
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        logger.error("Invalid date value: %r", value)
        return None


def _derive_phase(over_idx: int, is_super_over: bool) -> str:
    """Derive match phase from over index.

    Args:
        over_idx: 0-indexed over.
        is_super_over: True if over indicates a super over.

    Returns:
        Phase label.
    """
    if is_super_over:
        return "super_over"
    if 0 <= over_idx <= 5:
        return "powerplay"
    if 6 <= over_idx <= 14:
        return "middle"
    return "death"


def _extract_people_registry(info: dict[str, Any]) -> dict[str, str]:
    """Read ``info.registry.people`` name → id map, or empty dict if absent.

    Args:
        info: Cricsheet ``info`` object.

    Returns:
        Mapping of exact JSON display strings to hex people ids.
    """
    reg = info.get("registry")
    if not isinstance(reg, dict):
        return {}
    raw = reg.get("people")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for name_key, pid in raw.items():
        if not isinstance(name_key, str):
            continue
        if not isinstance(pid, str) or not pid.strip():
            continue
        out[name_key] = pid.strip()
    return out


def _people_id(people_map: dict[str, str], name: Any) -> str | None:
    """Resolve a people id for a delivery or wicket name cell.

    Args:
        people_map: Registry map for this match file.
        name: Cell value from JSON (string or null).

    Returns:
        Hex id string when present in ``people_map``, else None.
    """
    if name is None:
        return None
    if isinstance(name, float) and pd.isna(name):
        return None
    key = str(name).strip()
    if not key:
        return None
    return people_map.get(key)


def _fielder_name_cell(fielder: Any) -> str | None:
    """Extract fielder display name from a fielders[] entry.

    Args:
        fielder: One element of ``wickets[].fielders``.

    Returns:
        Name string or None.
    """
    if isinstance(fielder, dict):
        return fielder.get("name")
    return None


def _fielder_sub_cell(fielder: Any) -> bool | None:
    """Extract substitute flag from a fielders[] entry.

    Args:
        fielder: One element of ``wickets[].fielders``.

    Returns:
        Substitute flag or None when absent.
    """
    if isinstance(fielder, dict):
        return bool(fielder.get("substitute")) if "substitute" in fielder else None
    return None


def _parse_wicket(wicket: Any, people_map: dict[str, str]) -> dict[str, Any]:
    """Parse a single wicket dict into flattened fields and people ids.

    Args:
        wicket: Wicket object from JSON (or None).
        people_map: Registry map for this match file.

    Returns:
        Flattened wicket fields with None when absent.
    """
    empty = {
        "wicket_player_out": None,
        "wicket_kind": None,
        "wicket_fielder1": None,
        "wicket_fielder2": None,
        "fielder1_is_sub": None,
        "fielder2_is_sub": None,
        "wicket_player_out_id": None,
        "wicket_fielder1_id": None,
        "wicket_fielder2_id": None,
    }
    if not isinstance(wicket, dict):
        return empty

    fielders = wicket.get("fielders", [])
    f1 = fielders[0] if isinstance(fielders, list) and len(fielders) >= 1 else None
    f2 = fielders[1] if isinstance(fielders, list) and len(fielders) >= 2 else None
    po = wicket.get("player_out")
    fn1 = _fielder_name_cell(f1)
    fn2 = _fielder_name_cell(f2)
    base = {
        "wicket_player_out": po,
        "wicket_kind": wicket.get("kind"),
        "wicket_fielder1": fn1,
        "wicket_fielder2": fn2,
        "fielder1_is_sub": _fielder_sub_cell(f1),
        "fielder2_is_sub": _fielder_sub_cell(f2),
        "wicket_player_out_id": _people_id(people_map, po),
        "wicket_fielder1_id": _people_id(people_map, fn1),
        "wicket_fielder2_id": _people_id(people_map, fn2),
    }
    return base


def _prefix_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Prefix keys for second wicket output.

    Args:
        d: Parsed wicket dict (first-wicket schema).
        prefix: Prefix to apply to all keys.

    Returns:
        New dict with prefixed keys.
    """
    return {f"{prefix}{k.removeprefix('wicket_')}": v for k, v in d.items()}

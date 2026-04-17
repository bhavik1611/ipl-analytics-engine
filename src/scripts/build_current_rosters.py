"""Build team–player roster reference files from latest-season match deliveries.

Reads per-match parquet files, finds the maximum ``season`` value, then collects
distinct (team, player) pairs where the player batted or bowled for that team
in that season. If ``--json-out`` already exists (e.g. hand-edited roles), those
``role`` / ``is_captain`` values are **merged** for matching (team, player) rows
before overwrite. JSON uses ``role`` as a list of strings; CSV stores the same
roles joined with ``"; "``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import get_project_paths, load_env

logger = logging.getLogger(__name__)

_SOURCE = "derived_match_deliveries_latest_season"
_CSV_COLUMNS = [
    "team",
    "player",
    "role",
    "is_captain",
    "season",
    "source",
    "as_of_date",
]


def _coerce_role_list(value: object) -> list[str]:
    """Normalize JSON ``role`` (list or legacy string) to non-empty strings."""

    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def _role_list_to_csv_cell(roles: list[str]) -> str:
    """Serialize role tags for the flat roster CSV."""

    return "; ".join(roles)


def load_roster_overrides(path: Path) -> dict[tuple[str, str], tuple[list[str], bool]]:
    """Load (team, player) → (role list, is_captain) from a roster JSON file.

    Args:
        path: Path to ``current_rosters``-style JSON.

    Returns:
        Override map; empty if the file is missing or invalid.
    """

    out: dict[tuple[str, str], tuple[list[str], bool]] = {}
    if not path.is_file():
        return out
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Roster override read failed path=%s err=%s", path, exc)
        return out
    teams = raw.get("teams")
    if not isinstance(teams, dict):
        return out
    for team_name, players in teams.items():
        t = str(team_name).strip()
        if not t or not isinstance(players, list):
            continue
        for cell in players:
            if not isinstance(cell, dict):
                continue
            name = str(cell.get("name", "")).strip()
            if not name:
                continue
            roles = _coerce_role_list(cell.get("role"))
            cap = bool(cell.get("is_captain", False))
            out[(t, name)] = (roles, cap)
    return out


def _csv_row_for_player(
    team: str,
    cell: dict[str, object],
    season: int,
    source: str,
    as_of: str,
) -> dict[str, object] | None:
    """Build one CSV row dict for a player entry or None if invalid."""

    name = str(cell.get("name", "")).strip()
    if not name:
        return None
    roles = _coerce_role_list(cell.get("role"))
    return {
        "team": team,
        "player": name,
        "role": _role_list_to_csv_cell(roles),
        "is_captain": bool(cell.get("is_captain", False)),
        "season": season,
        "source": source,
        "as_of_date": as_of,
    }


def roster_csv_rows_from_json_document(doc: dict[str, object]) -> list[dict[str, object]]:
    """Flatten a roster JSON document to CSV-ready row dicts.

    Args:
        doc: Parsed JSON with ``season``, ``source``, ``as_of_date``, ``teams``.

    Returns:
        One dict per player with ``_CSV_COLUMNS`` keys.
    """

    season = int(doc.get("season", 0) or 0)
    source = str(doc.get("source", _SOURCE) or _SOURCE)
    as_of = str(doc.get("as_of_date", date.today().isoformat()))
    teams = doc.get("teams")
    rows: list[dict[str, object]] = []
    if not isinstance(teams, dict):
        return rows
    for team_name in sorted(teams.keys()):
        players = teams[team_name]
        if not isinstance(players, list):
            continue
        t = str(team_name).strip()
        for cell in sorted(players, key=lambda x: str((x or {}).get("name", ""))):
            if not isinstance(cell, dict):
                continue
            row = _csv_row_for_player(t, cell, season, source, as_of)
            if row is not None:
                rows.append(row)
    return rows


def list_parquet_paths(matches_dir: Path) -> list[Path]:
    """Return sorted parquet paths under ``matches_dir``."""

    return sorted(matches_dir.glob("*.parquet"))


def _list_parquet_paths(matches_dir: Path) -> list[Path]:
    """Backward-compatible alias for older callers/tests."""

    return list_parquet_paths(matches_dir)


def max_season_from_parquets(paths: list[Path]) -> int | None:
    """Scan parquet files for the highest ``season`` value.

    Args:
        paths: Per-match parquet paths.

    Returns:
        Maximum season, or None if no readable seasons.
    """

    if not paths:
        return None
    peak: int | None = None
    for path in paths:
        df = pd.read_parquet(path, columns=["season"])
        local_max = int(df["season"].max())
        peak = local_max if peak is None else max(peak, local_max)
    return peak


def _max_season_from_parquets(paths: list[Path]) -> int | None:
    """Backward-compatible alias for older callers/tests."""

    return max_season_from_parquets(paths)


def pairs_for_season(paths: list[Path], season: int) -> set[tuple[str, str]]:
    """Collect (team, player) for batters and bowlers in the given season.

    Args:
        paths: Per-match parquet paths.
        season: Season year to filter.

    Returns:
        Set of (team_name, player_name) tuples.
    """

    cols = ["season", "batter", "bowler", "batting_team", "bowling_team"]
    chunks: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path, columns=cols)
        sub = df.loc[df["season"] == season]
        if sub.empty:
            continue
        bat = sub.assign(team=sub["batting_team"], player=sub["batter"])[["team", "player"]]
        bow = sub.assign(team=sub["bowling_team"], player=sub["bowler"])[["team", "player"]]
        chunks.append(bat)
        chunks.append(bow)
    if not chunks:
        return set()
    merged = pd.concat(chunks, ignore_index=True)
    merged = merged.dropna(subset=["team", "player"])
    merged["team"] = merged["team"].astype(str).str.strip()
    merged["player"] = merged["player"].astype(str).str.strip()
    merged = merged.loc[(merged["team"] != "") & (merged["player"] != "")]
    pairs = set(zip(merged["team"].tolist(), merged["player"].tolist()))
    return pairs


def _pairs_for_season(paths: list[Path], season: int) -> set[tuple[str, str]]:
    """Backward-compatible alias for older callers/tests."""

    return pairs_for_season(paths, season)


def _pairs_to_csv_rows(
    pairs: set[tuple[str, str]],
    season: int,
    as_of: str,
    overrides: dict[tuple[str, str], tuple[list[str], bool]],
) -> list[dict[str, object]]:
    """Sort pairs and build CSV row dicts."""

    rows: list[dict[str, object]] = []
    for team, player in sorted(pairs):
        roles, cap = overrides.get((team, player), ([], False))
        rows.append(
            {
                "team": team,
                "player": player,
                "role": _role_list_to_csv_cell(roles),
                "is_captain": cap,
                "season": season,
                "source": _SOURCE,
                "as_of_date": as_of,
            }
        )
    return rows


def _pairs_to_json_doc(
    pairs: set[tuple[str, str]],
    season: int,
    as_of: str,
    overrides: dict[tuple[str, str], tuple[list[str], bool]],
) -> dict[str, object]:
    """Nest players under team keys for JSON export."""

    teams: dict[str, list[dict[str, object]]] = {}
    for team, player in sorted(pairs):
        roles, cap = overrides.get((team, player), ([], False))
        teams.setdefault(team, []).append({"name": player, "role": roles, "is_captain": cap})
    return {
        "season": season,
        "as_of_date": as_of,
        "source": _SOURCE,
        "teams": teams,
    }


def write_roster_outputs(
    csv_path: Path,
    json_path: Path,
    pairs: set[tuple[str, str]],
    season: int,
) -> None:
    """Write CSV and JSON roster artifacts, merging prior JSON when present."""

    as_of = date.today().isoformat()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    overrides = load_roster_overrides(json_path) if pairs and json_path.is_file() else {}
    if overrides:
        logger.info("Merged %d roster override entries from %s", len(overrides), json_path)
    if not pairs:
        pd.DataFrame(columns=_CSV_COLUMNS).to_csv(csv_path, index=False)
        empty_doc: dict[str, object] = {
            "season": season,
            "as_of_date": as_of,
            "source": _SOURCE,
            "teams": {},
        }
        json_path.write_text(json.dumps(empty_doc, indent=2), encoding="utf-8")
        return
    rows = _pairs_to_csv_rows(pairs, season, as_of, overrides)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    doc = _pairs_to_json_doc(pairs, season, as_of, overrides)
    json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _write_outputs(csv_path: Path, json_path: Path, pairs: set[tuple[str, str]], season: int) -> None:
    """Backward-compatible alias for older callers/tests."""

    write_roster_outputs(csv_path, json_path, pairs, season)


def build_current_rosters(*, matches_dir: Path, csv_out: Path, json_out: Path) -> int:
    """Build current roster CSV/JSON from match parquet files.

    Args:
        matches_dir: Directory containing match parquet files.
        csv_out: Output path for roster CSV.
        json_out: Output path for roster JSON.

    Returns:
        Exit-style status code: 0 on success.
    """

    paths = list_parquet_paths(matches_dir)
    if not paths:
        logger.warning("No parquet files under %s; writing empty roster files.", matches_dir)
        write_roster_outputs(csv_out, json_out, set(), 0)
        return 0
    peak = max_season_from_parquets(paths)
    if peak is None:
        logger.warning("Could not determine season; writing empty roster files.")
        write_roster_outputs(csv_out, json_out, set(), 0)
        return 0
    pairs = pairs_for_season(paths, peak)
    write_roster_outputs(csv_out, json_out, pairs, peak)
    logger.info(
        "Wrote %d roster rows for season=%d to %s and %s",
        len(pairs),
        peak,
        csv_out,
        json_out,
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for roster build."""

    p = argparse.ArgumentParser(description="Build current roster CSV/JSON from match parquets.")
    p.add_argument(
        "--matches-dir",
        default="",
        help="Override matches directory (default: PROCESSED_DIR/matches from env)",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=Path("data/reference/current_rosters.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=Path("data/reference/current_rosters.json"),
        help="Output JSON path",
    )
    return p


def main() -> int:
    """Entry point for roster generation."""

    load_env()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()
    paths_obj = get_project_paths()
    matches_dir = Path(args.matches_dir) if args.matches_dir else paths_obj.matches_dir
    return build_current_rosters(matches_dir=matches_dir, csv_out=args.csv_out, json_out=args.json_out)


if __name__ == "__main__":
    raise SystemExit(main())

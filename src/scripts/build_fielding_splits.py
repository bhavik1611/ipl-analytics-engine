"""Build fielding split tables by venue and by season.

Outputs:
  - data/processed/fielding_venue_splits.csv
  - data/processed/fielding_season_splits.csv

These are used to enrich static matchup JSONs with fielding broken down into:
overall career, venue-wise, season-wise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import get_project_paths, load_env
from src.scoring.calculator import calculate_match_points
from src.utils.aggregator import _with_normalized_venues  # noqa: SLF001

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class _OutPaths:
    venue_csv: Path
    season_csv: Path


_NEEDED_WICKET_COLS: list[str] = [
    "match_id",
    "season",
    "date",
    "venue",
    "is_super_over",
    "is_wicket",
    "wicket_kind",
    "wicket_fielder1",
    "wicket_fielder2",
    "fielder1_is_sub",
    "fielder2_is_sub",
    "wicket2_kind",
    "wicket2_fielder1",
    "wicket2_fielder2",
    "wicket2_fielder1_is_sub",
    "wicket2_fielder2_is_sub",
]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--venue-out",
        type=Path,
        default=_REPO_ROOT / "data/processed/fielding_venue_splits.csv",
    )
    p.add_argument(
        "--season-out",
        type=Path,
        default=_REPO_ROOT / "data/processed/fielding_season_splits.csv",
    )
    return p.parse_args(argv)


def _is_present_name(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def _is_sub_flag(value: object) -> bool:
    if value is None:
        return False
    return bool(value)


def _credit_fielder(name: object, is_sub: object) -> str | None:
    if not _is_present_name(name):
        return None
    if _is_sub_flag(is_sub):
        return None
    return str(name)


def _rows_for_wicket(base: dict[str, Any], r: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kind = r.get("wicket_kind")
    if kind in {"caught", "caught and bowled"}:
        p = _credit_fielder(r.get("wicket_fielder1"), r.get("fielder1_is_sub"))
        if p:
            rows.append({**base, "player": p, "catches": 1, "runouts": 0, "stumpings": 0})
    if kind == "stumped":
        p = _credit_fielder(r.get("wicket_fielder1"), r.get("fielder1_is_sub"))
        if p:
            rows.append({**base, "player": p, "catches": 0, "runouts": 0, "stumpings": 1})
    if kind == "run out":
        p1 = _credit_fielder(r.get("wicket_fielder1"), r.get("fielder1_is_sub"))
        p2 = _credit_fielder(r.get("wicket_fielder2"), r.get("fielder2_is_sub"))
        if p1:
            rows.append({**base, "player": p1, "catches": 0, "runouts": 1, "stumpings": 0})
        if p2 and p2 != p1:
            rows.append({**base, "player": p2, "catches": 0, "runouts": 1, "stumpings": 0})
    return rows


def _rows_for_wicket2(base: dict[str, Any], r: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kind2 = r.get("wicket2_kind")
    if kind2 != "run out":
        return rows
    p1 = _credit_fielder(r.get("wicket2_fielder1"), r.get("wicket2_fielder1_is_sub"))
    p2 = _credit_fielder(r.get("wicket2_fielder2"), r.get("wicket2_fielder2_is_sub"))
    if p1:
        rows.append({**base, "player": p1, "catches": 0, "runouts": 1, "stumpings": 0})
    if p2 and p2 != p1:
        rows.append({**base, "player": p2, "catches": 0, "runouts": 1, "stumpings": 0})
    return rows


def _fielding_event_rows(match_df: pd.DataFrame) -> pd.DataFrame:
    df = _with_normalized_venues(match_df).loc[~match_df["is_super_over"].fillna(False).astype(bool)].copy()
    wk = df.loc[df["is_wicket"].fillna(False).astype(bool)].copy()
    if wk.empty:
        return pd.DataFrame(columns=["match_id", "season", "venue", "player", "catches", "runouts", "stumpings"])
    meta = df.iloc[0]
    base = {"match_id": str(meta["match_id"]), "season": int(meta["season"]), "venue": str(meta["venue"])}
    out_rows: list[dict[str, Any]] = []
    for _, r in wk.iterrows():
        out_rows.extend(_rows_for_wicket(base, r))
        out_rows.extend(_rows_for_wicket2(base, r))
    return pd.DataFrame(out_rows)


def _fielding_points_rows(match_df: pd.DataFrame) -> pd.DataFrame:
    pts = calculate_match_points(_with_normalized_venues(match_df))
    if pts.empty:
        return pd.DataFrame(columns=["match_id", "season", "venue", "player", "fantasy_fielding"])
    return pts.loc[:, ["match_id", "season", "venue", "player", "fantasy_fielding"]].copy()


def _combine_parts(parts: list[pd.DataFrame], cols: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=cols)
    x = pd.concat(parts, ignore_index=True)
    if x.empty:
        return pd.DataFrame(columns=cols)
    return x


def build_splits(matches_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan match parquets once and build venue/season split tables."""

    event_parts: list[pd.DataFrame] = []
    fp_parts: list[pd.DataFrame] = []
    files = sorted(matches_dir.glob("*.parquet"))
    it = tqdm(
        files,
        total=len(files),
        desc="Building fielding splits",
        unit="match",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for path in it:
        match_df = pd.read_parquet(path)
        event_parts.append(_fielding_event_rows(match_df.loc[:, _NEEDED_WICKET_COLS]))
        fp_parts.append(_fielding_points_rows(match_df))

    events = _combine_parts(event_parts, ["match_id", "season", "venue", "player", "catches", "runouts", "stumpings"])
    points = _combine_parts(fp_parts, ["match_id", "season", "venue", "player", "fantasy_fielding"])

    # Aggregate counts
    by_venue_counts = events.groupby(["player", "venue"], as_index=False).agg(
        matches=("match_id", "nunique"),
        catches=("catches", "sum"),
        runouts=("runouts", "sum"),
        stumpings=("stumpings", "sum"),
    )
    by_season_counts = events.groupby(["player", "season"], as_index=False).agg(
        matches=("match_id", "nunique"),
        catches=("catches", "sum"),
        runouts=("runouts", "sum"),
        stumpings=("stumpings", "sum"),
    )

    # Aggregate fantasy fielding
    by_venue_fp = points.groupby(["player", "venue"], as_index=False).agg(
        fantasy_fielding_total=("fantasy_fielding", "sum"),
        fantasy_fielding_avg=("fantasy_fielding", "mean"),
    )
    by_season_fp = points.groupby(["player", "season"], as_index=False).agg(
        fantasy_fielding_total=("fantasy_fielding", "sum"),
        fantasy_fielding_avg=("fantasy_fielding", "mean"),
    )

    venue = by_venue_counts.merge(by_venue_fp, on=["player", "venue"], how="left").fillna(
        {"fantasy_fielding_total": 0, "fantasy_fielding_avg": 0.0}
    )
    season = by_season_counts.merge(by_season_fp, on=["player", "season"], how="left").fillna(
        {"fantasy_fielding_total": 0, "fantasy_fielding_avg": 0.0}
    )
    venue = venue.sort_values(["player", "venue"]).reset_index(drop=True)
    season = season.sort_values(["player", "season"]).reset_index(drop=True)
    return venue, season


def main(argv: list[str] | None = None) -> int:
    load_env()
    paths = get_project_paths()
    ns = _parse_args(argv)
    out = _OutPaths(venue_csv=ns.venue_out, season_csv=ns.season_out)
    out.venue_csv.parent.mkdir(parents=True, exist_ok=True)
    out.season_csv.parent.mkdir(parents=True, exist_ok=True)
    venue, season = build_splits(paths.matches_dir)
    venue.to_csv(out.venue_csv, index=False)
    season.to_csv(out.season_csv, index=False)
    print(f"Wrote fielding venue splits: {out.venue_csv} rows={len(venue)}")
    print(f"Wrote fielding season splits: {out.season_csv} rows={len(season)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


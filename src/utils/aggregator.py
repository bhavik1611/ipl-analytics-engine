"""Phase 3 aggregation utilities.

This module consumes per-match deliveries parquet files, calls the scoring
calculator for fantasy points, and emits analytical aggregated CSVs.

Career tables remain keyed by display ``player`` name and add ``player_id``
(Cricsheet ``registry.people`` hex) when present on deliveries for stable joins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.scoring.calculator import calculate_match_points
from src.utils.logging_support import ensure_pipeline_logger, new_run_id
from src.utils.parser import PEOPLE_ID_DELIVERY_COLUMNS

logger = logging.getLogger(__name__)


def _progress_milestone_indices(total: int) -> set[int]:
    """Return 10%–100% file indices (inclusive) for coarse progress logging.

    Args:
        total: Number of parquet files to process.

    Returns:
        Set of 1-based indices at which to emit a progress milestone log.
    """

    if total <= 0:
        return set()
    out: set[int] = set()
    for pct in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        out.add(max(1, (total * pct + 99) // 100))
    out.add(total)
    return out


@dataclass(frozen=True)
class _AggPaths:
    career_batting: Path
    career_bowling: Path
    career_fielding: Path
    venue_splits: Path
    phase_splits: Path
    season_trends: Path


def _output_paths(output_dir: Path) -> _AggPaths:
    return _AggPaths(
        career_batting=output_dir / "career_batting.csv",
        career_bowling=output_dir / "career_bowling.csv",
        career_fielding=output_dir / "career_fielding.csv",
        venue_splits=output_dir / "venue_splits.csv",
        phase_splits=output_dir / "phase_splits.csv",
        season_trends=output_dir / "season_trends.csv",
    )


def _all_outputs_exist(paths: _AggPaths) -> bool:
    return all(p.exists() for p in paths.__dict__.values())


def _safe_div(numer: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return float(numer) / float(denom)


def _ensure_delivery_people_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add nullable people-id columns when missing (older match parquets).

    Args:
        df: Per-match deliveries read from parquet.

    Returns:
        DataFrame with ``PEOPLE_ID_DELIVERY_COLUMNS`` present.
    """
    out = df.copy()
    for col in PEOPLE_ID_DELIVERY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _series_mode_id(series: pd.Series) -> object:
    """Pick a single canonical id string from a series (mode, else first).

    Args:
        series: Id cells for one grouped entity (may be all NA).

    Returns:
        A non-empty string id, or ``pd.NA`` when none exist.
    """
    valid = series.dropna()
    if valid.empty:
        return pd.NA
    as_str = valid.astype(str).str.strip()
    as_str = as_str[as_str != ""]
    if as_str.empty:
        return pd.NA
    mode = as_str.mode()
    if len(mode) > 0:
        return str(mode.iloc[0])
    return str(as_str.iloc[0])


def _cell_people_id(value: object) -> object:
    """Normalize a single registry id cell for aggregation rows.

    Args:
        value: Raw id from a delivery row (may be NA).

    Returns:
        Stripped string id or ``pd.NA``.
    """
    if value is None:
        return pd.NA
    if isinstance(value, float) and pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _list_parquets(processed_dir: Path) -> list[Path]:
    """List match parquet paths; prefer ``dir/*.parquet`` else ``dir/matches/*.parquet``."""

    direct = sorted(processed_dir.glob("*.parquet"))
    if direct:
        return direct
    nested = processed_dir / "matches"
    if nested.is_dir():
        return sorted(nested.glob("*.parquet"))
    return []


def _normalize_venue_name(venue: object) -> str:
    """Map raw Cricsheet venue strings to a canonical label for aggregation.

    IPL data often repeats the same ground with a trailing city suffix, for
    example ``Wankhede Stadium`` vs ``Wankhede Stadium, Mumbai``. We keep the
    primary venue name (text before the first comma) so splits merge correctly.

    Args:
        venue: Raw venue cell (may be NaN or non-string).

    Returns:
        Normalized non-empty venue string, or the literal ``unknown`` if missing.
    """

    if venue is None or (isinstance(venue, float) and pd.isna(venue)):
        return "unknown"
    text = str(venue).strip()
    if not text:
        return "unknown"
    primary = text.split(",", maxsplit=1)[0].strip()
    if not primary:
        return "unknown"
    return " ".join(primary.split())


def _with_normalized_venues(match_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the match frame with ``venue`` normalized.

    Args:
        match_df: Per-match deliveries DataFrame.

    Returns:
        DataFrame with ``venue`` rewritten for downstream scoring and splits.
    """

    out = match_df.copy()
    if "venue" in out.columns:
        out["venue"] = out["venue"].apply(_normalize_venue_name)
    return out


def _filter_non_super(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.loc[~df["is_super_over"]].copy()  # noqa: E712


def _batting_innings_rows(match_df: pd.DataFrame) -> pd.DataFrame:
    df = _filter_non_super(match_df)
    bdf = df.loc[df["batter"].notna()].copy()
    if bdf.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "season",
                "date",
                "venue",
                "player",
                "player_id",
                "runs",
                "balls",
                "fours",
                "sixes",
                "dismissed",
            ]
        )
    bdf["player"] = bdf["batter"].astype(str)
    grp = bdf.groupby(["match_id", "season", "date", "venue", "player"], as_index=False)
    out = grp.agg(
        runs=("runs_batter", "sum"),
        balls=("is_legal_delivery", lambda s: int(s.astype(bool).sum())),
        fours=("is_four", lambda s: int(s.astype(bool).sum())),
        sixes=("is_six", lambda s: int(s.astype(bool).sum())),
        player_id=("batter_id", _series_mode_id),
    )
    wk = df.loc[df["is_wicket"]].copy()
    if wk.empty:
        out["dismissed"] = 0
        return out
    wk_players = pd.concat(
        [
            wk.loc[
                wk["wicket_player_out"].notna(),
                ["match_id", "wicket_player_out", "wicket_player_out_id"],
            ].rename(
                columns={"wicket_player_out": "player", "wicket_player_out_id": "dismiss_id"}
            ),
            wk.loc[
                wk["wicket2_player_out"].notna(),
                ["match_id", "wicket2_player_out", "wicket2_player_out_id"],
            ].rename(
                columns={"wicket2_player_out": "player", "wicket2_player_out_id": "dismiss_id"}
            ),
        ],
        ignore_index=True,
    )
    wk_players["player"] = wk_players["player"].astype(str)
    wk_players["dismissed"] = 1
    dismiss = wk_players.groupby(["match_id", "player"], as_index=False).agg(
        dismissed=("dismissed", "sum"),
        dismiss_player_id=("dismiss_id", _series_mode_id),
    )
    out = out.merge(dismiss, on=["match_id", "player"], how="left")
    out["player_id"] = out["player_id"].where(out["player_id"].notna(), out["dismiss_player_id"])
    out = out.drop(columns=["dismiss_player_id"])
    out["dismissed"] = out["dismissed"].fillna(0).astype(int)
    return out


def _batting_career_from_innings(innings: pd.DataFrame) -> pd.DataFrame:
    if innings.empty:
        cols = [
            "player",
            "player_id",
            "matches",
            "innings",
            "runs_total",
            "balls_faced",
            "fours",
            "sixes",
            "dismissals",
            "average",
            "strike_rate",
            "not_outs",
            "ducks",
            "fifties",
            "seventyfives",
            "hundreds",
            "fantasy_batting_total",
            "fantasy_batting_avg",
            "fantasy_batting_max",
            "fantasy_batting_min",
        ]
        return pd.DataFrame(columns=cols)

    per_player = innings.groupby("player", as_index=False).agg(
        matches=("match_id", "nunique"),
        innings=("match_id", "count"),
        runs_total=("runs", "sum"),
        balls_faced=("balls", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        dismissals=("dismissed", "sum"),
        player_id=("player_id", _series_mode_id),
    )
    per_player["average"] = per_player.apply(
        lambda r: _safe_div(r["runs_total"], r["dismissals"]), axis=1
    )
    per_player["strike_rate"] = per_player.apply(
        lambda r: _safe_div(r["runs_total"] * 100.0, r["balls_faced"]), axis=1
    )
    per_player["not_outs"] = per_player["innings"] - per_player["dismissals"]

    def _duck(row: pd.Series) -> int:
        return int((row["runs"] == 0) and (row["balls"] > 0) and (row["dismissed"] > 0))

    innings["is_duck"] = innings.apply(_duck, axis=1)
    innings["is_50"] = innings["runs"].between(50, 74)
    innings["is_75"] = innings["runs"].between(75, 99)
    innings["is_100"] = innings["runs"] >= 100
    milestones = innings.groupby("player", as_index=False).agg(
        ducks=("is_duck", "sum"),
        fifties=("is_50", "sum"),
        seventyfives=("is_75", "sum"),
        hundreds=("is_100", "sum"),
    )
    out = per_player.merge(milestones, on="player", how="left")
    return out


def _bowling_innings_rows(match_df: pd.DataFrame) -> pd.DataFrame:
    df = _filter_non_super(match_df)
    b = df.loc[df["bowler"].notna()].copy()
    if b.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "season",
                "date",
                "venue",
                "player",
                "player_id",
                "legal_deliveries",
                "runs_conceded",
                "wickets",
                "maidens",
            ]
        )
    b["player"] = b["bowler"].astype(str)
    grp = b.groupby(["match_id", "season", "date", "venue", "player"], as_index=False)
    out = grp.agg(
        legal_deliveries=("is_legal_delivery", lambda s: int(s.astype(bool).sum())),
        runs_conceded=("runs_bowler", "sum"),
        wickets=("is_bowler_wicket", lambda s: int(s.astype(bool).sum())),
        player_id=("bowler_id", _series_mode_id),
    )

    over_stats = b.groupby(
        ["match_id", "player", "innings_num", "over_1indexed"], sort=False
    ).agg(
        legal_balls=(
            "is_legal_delivery",
            lambda s: int(s.astype(bool).sum()),
        ),
        wide_any=("is_wide", "any"),
        noball_any=("is_noball", "any"),
        runs_max=("runs_batter", "max"),
    )
    maiden_df = over_stats.reset_index()
    maiden_df["is_maiden"] = (
        (maiden_df["legal_balls"] == 6)
        & (~maiden_df["wide_any"].fillna(False).astype(bool))
        & (~maiden_df["noball_any"].fillna(False).astype(bool))
        & (maiden_df["runs_max"] == 0)
    )
    maidens = (
        maiden_df.loc[maiden_df["is_maiden"]]
        .groupby(["match_id", "player"], as_index=False)
        .size()
    )
    maidens = maidens.rename(columns={"size": "maidens"})
    out = out.merge(maidens, on=["match_id", "player"], how="left")
    out["maidens"] = out["maidens"].fillna(0).astype(int)
    return out


def _bowling_career_from_innings(innings: pd.DataFrame) -> pd.DataFrame:
    if innings.empty:
        cols = [
            "player",
            "player_id",
            "matches",
            "innings_bowled",
            "legal_deliveries",
            "overs_bowled",
            "runs_conceded",
            "wickets",
            "economy",
            "bowling_average",
            "maidens",
            "four_wicket_hauls",
            "fantasy_bowling_total",
            "fantasy_bowling_avg",
            "fantasy_bowling_max",
            "fantasy_bowling_min",
        ]
        return pd.DataFrame(columns=cols)

    out = innings.groupby("player", as_index=False).agg(
        matches=("match_id", "nunique"),
        innings_bowled=("match_id", lambda s: int((s.notna()).sum())),
        legal_deliveries=("legal_deliveries", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        wickets=("wickets", "sum"),
        maidens=("maidens", "sum"),
        player_id=("player_id", _series_mode_id),
    )
    out["overs_bowled"] = out["legal_deliveries"] / 6.0
    out["economy"] = out.apply(
        lambda r: _safe_div(r["runs_conceded"], r["overs_bowled"]), axis=1
    )
    out["bowling_average"] = out.apply(
        lambda r: _safe_div(r["runs_conceded"], r["wickets"]), axis=1
    )

    per_match = innings.groupby(["match_id", "player"], as_index=False).agg(
        w=("wickets", "sum")
    )
    fw = (
        per_match.loc[per_match["w"] >= 4]
        .groupby("player", as_index=False)
        .size()
        .rename(columns={"size": "four_wicket_hauls"})
    )
    out = out.merge(fw, on="player", how="left")
    out["four_wicket_hauls"] = out["four_wicket_hauls"].fillna(0).astype(int)
    return out


def _fielding_match_rows(match_df: pd.DataFrame) -> pd.DataFrame:
    df = _filter_non_super(match_df)
    wk = df.loc[df["is_wicket"]].copy()
    if wk.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "season",
                "date",
                "venue",
                "player",
                "player_id",
                "catches",
                "runouts",
                "stumpings",
            ]
        )

    def _non_sub_name(name: object, flag: object) -> str | None:
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return None
        if isinstance(name, str) and name.strip() == "":
            return None
        if flag is True:
            return None
        return str(name)

    rows: list[dict[str, object]] = []
    meta = df.iloc[0]
    base = {
        "match_id": str(meta["match_id"]),
        "season": int(meta["season"]),
        "date": meta["date"],
        "venue": str(meta["venue"]),
    }

    for _, r in wk.iterrows():
        kind = r.get("wicket_kind")
        if kind in {"caught", "caught and bowled"}:
            p = _non_sub_name(r.get("wicket_fielder1"), r.get("fielder1_is_sub"))
            if p:
                rows.append(
                    {
                        **base,
                        "player": p,
                        "player_id": _cell_people_id(r.get("wicket_fielder1_id")),
                        "catches": 1,
                        "runouts": 0,
                        "stumpings": 0,
                    }
                )
        if kind == "stumped":
            p = _non_sub_name(r.get("wicket_fielder1"), r.get("fielder1_is_sub"))
            if p:
                rows.append(
                    {
                        **base,
                        "player": p,
                        "player_id": _cell_people_id(r.get("wicket_fielder1_id")),
                        "catches": 0,
                        "runouts": 0,
                        "stumpings": 1,
                    }
                )
        if kind == "run out":
            p1 = _non_sub_name(r.get("wicket_fielder1"), r.get("fielder1_is_sub"))
            p2 = _non_sub_name(r.get("wicket_fielder2"), r.get("fielder2_is_sub"))
            if p1:
                rows.append(
                    {
                        **base,
                        "player": p1,
                        "player_id": _cell_people_id(r.get("wicket_fielder1_id")),
                        "catches": 0,
                        "runouts": 1,
                        "stumpings": 0,
                    }
                )
            if p2 and p2 != p1:
                rows.append(
                    {
                        **base,
                        "player": p2,
                        "player_id": _cell_people_id(r.get("wicket_fielder2_id")),
                        "catches": 0,
                        "runouts": 1,
                        "stumpings": 0,
                    }
                )

        kind2 = r.get("wicket2_kind")
        if kind2 == "run out":
            p1 = _non_sub_name(
                r.get("wicket2_fielder1"), r.get("wicket2_fielder1_is_sub")
            )
            p2 = _non_sub_name(
                r.get("wicket2_fielder2"), r.get("wicket2_fielder2_is_sub")
            )
            if p1:
                rows.append(
                    {
                        **base,
                        "player": p1,
                        "player_id": _cell_people_id(r.get("wicket2_fielder1_id")),
                        "catches": 0,
                        "runouts": 1,
                        "stumpings": 0,
                    }
                )
            if p2 and p2 != p1:
                rows.append(
                    {
                        **base,
                        "player": p2,
                        "player_id": _cell_people_id(r.get("wicket2_fielder2_id")),
                        "catches": 0,
                        "runouts": 1,
                        "stumpings": 0,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "season",
                "date",
                "venue",
                "player",
                "player_id",
                "catches",
                "runouts",
                "stumpings",
            ]
        )
    return pd.DataFrame(rows)


def _phase_label(over_1indexed: int) -> str:
    if 1 <= over_1indexed <= 6:
        return "powerplay"
    if 7 <= over_1indexed <= 15:
        return "middle"
    return "death"


def _phase_splits_delivery_metrics(
    match_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _filter_non_super(match_df)
    if df.empty:
        bat_cols = [
            "match_id",
            "player",
            "phase",
            "runs_total",
            "balls_faced",
            "dot_balls",
        ]
        bowl_cols = [
            "match_id",
            "player",
            "phase",
            "runs_conceded",
            "legal_deliveries",
            "wickets",
        ]
        return pd.DataFrame(columns=bat_cols), pd.DataFrame(columns=bowl_cols)

    df = df.copy()
    df["phase3"] = df["over_1indexed"].astype(int).apply(_phase_label)

    b = df.loc[df["batter"].notna()].copy()
    b["player"] = b["batter"].astype(str)
    bat = (
        b.groupby(["match_id", "player", "phase3"], as_index=False)
        .agg(
            runs_total=("runs_batter", "sum"),
            balls_faced=("is_legal_delivery", lambda s: int(s.astype(bool).sum())),
            dot_balls=("is_dot_ball", lambda s: int(s.astype(bool).sum())),
        )
        .rename(columns={"phase3": "phase"})
    )

    bo = df.loc[df["bowler"].notna()].copy()
    bo["player"] = bo["bowler"].astype(str)
    bowl = (
        bo.groupby(["match_id", "player", "phase3"], as_index=False)
        .agg(
            runs_conceded=("runs_bowler", "sum"),
            legal_deliveries=("is_legal_delivery", lambda s: int(s.astype(bool).sum())),
            wickets=("is_bowler_wicket", lambda s: int(s.astype(bool).sum())),
        )
        .rename(columns={"phase3": "phase"})
    )
    return bat, bowl


def _venue_splits_delivery_metrics(
    match_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _filter_non_super(match_df)
    if df.empty:
        bat_cols = [
            "match_id",
            "player",
            "venue",
            "runs_total",
            "balls_faced",
            "dismissals",
        ]
        bowl_cols = [
            "match_id",
            "player",
            "venue",
            "runs_conceded",
            "legal_deliveries",
            "wickets",
        ]
        return pd.DataFrame(columns=bat_cols), pd.DataFrame(columns=bowl_cols)

    venue = str(df.iloc[0]["venue"])
    b_inn = _batting_innings_rows(df)
    bat = b_inn.loc[:, ["match_id", "player", "runs", "balls", "dismissed"]].copy()
    bat = bat.rename(
        columns={
            "runs": "runs_total",
            "balls": "balls_faced",
            "dismissed": "dismissals",
        }
    )
    bat["venue"] = venue

    bo_inn = _bowling_innings_rows(df)
    bowl = bo_inn.loc[
        :, ["match_id", "player", "runs_conceded", "legal_deliveries", "wickets"]
    ].copy()
    bowl["venue"] = venue
    return bat, bowl


def _season_delivery_metrics(
    match_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _filter_non_super(match_df)
    if df.empty:
        bat_cols = [
            "match_id",
            "player",
            "season",
            "runs_total",
            "balls_faced",
            "dismissals",
        ]
        bowl_cols = [
            "match_id",
            "player",
            "season",
            "runs_conceded",
            "legal_deliveries",
            "wickets",
        ]
        return pd.DataFrame(columns=bat_cols), pd.DataFrame(columns=bowl_cols)

    season = int(df.iloc[0]["season"])
    b_inn = _batting_innings_rows(df)
    bat = b_inn.loc[:, ["match_id", "player", "runs", "balls", "dismissed"]].copy()
    bat = bat.rename(
        columns={
            "runs": "runs_total",
            "balls": "balls_faced",
            "dismissed": "dismissals",
        }
    )
    bat["season"] = season

    bo_inn = _bowling_innings_rows(df)
    bowl = bo_inn.loc[
        :, ["match_id", "player", "runs_conceded", "legal_deliveries", "wickets"]
    ].copy()
    bowl["season"] = season
    return bat, bowl


def _derive_team_latest(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(columns=["player", "team_latest"])
    p = points.copy()
    p["date_dt"] = pd.to_datetime(p["date"], errors="coerce")
    p = p.sort_values(["player", "date_dt", "match_id"], ascending=[True, True, True])
    last = p.groupby("player", as_index=False).tail(1)
    return last.loc[:, ["player", "team"]].rename(columns={"team": "team_latest"})


def _add_fantasy_batting(career: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    if career.empty:
        career["fantasy_batting_total"] = []
        return career
    g = points.groupby("player", as_index=False).agg(
        fantasy_batting_total=("fantasy_batting", "sum"),
        fantasy_batting_avg=("fantasy_batting", "mean"),
        fantasy_batting_max=("fantasy_batting", "max"),
        fantasy_batting_min=("fantasy_batting", "min"),
    )
    return career.merge(g, on="player", how="left").fillna(
        {
            "fantasy_batting_total": 0,
            "fantasy_batting_avg": 0.0,
            "fantasy_batting_max": 0,
            "fantasy_batting_min": 0,
        }
    )


def _add_fantasy_bowling(career: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    if career.empty:
        career["fantasy_bowling_total"] = []
        return career
    g = points.groupby("player", as_index=False).agg(
        fantasy_bowling_total=("fantasy_bowling", "sum"),
        fantasy_bowling_avg=("fantasy_bowling", "mean"),
        fantasy_bowling_max=("fantasy_bowling", "max"),
        fantasy_bowling_min=("fantasy_bowling", "min"),
    )
    return career.merge(g, on="player", how="left").fillna(
        {
            "fantasy_bowling_total": 0,
            "fantasy_bowling_avg": 0.0,
            "fantasy_bowling_max": 0,
            "fantasy_bowling_min": 0,
        }
    )


def _add_fantasy_fielding(career: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    if career.empty:
        career = career.copy()
        career["fantasy_fielding_total"] = pd.Series(dtype="int64")
        career["fantasy_fielding_avg"] = pd.Series(dtype="float64")
        return career
    g = points.groupby("player", as_index=False).agg(
        fantasy_fielding_total=("fantasy_fielding", "sum"),
        fantasy_fielding_avg=("fantasy_fielding", "mean"),
    )
    return career.merge(g, on="player", how="left").fillna(
        {"fantasy_fielding_total": 0, "fantasy_fielding_avg": 0.0}
    )


def _career_fielding_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        cols = [
            "player",
            "player_id",
            "matches",
            "catches",
            "runouts",
            "stumpings",
            "fantasy_fielding_total",
            "fantasy_fielding_avg",
        ]
        return pd.DataFrame(columns=cols)
    return rows.groupby("player", as_index=False).agg(
        matches=("match_id", "nunique"),
        catches=("catches", "sum"),
        runouts=("runouts", "sum"),
        stumpings=("stumpings", "sum"),
        player_id=("player_id", _series_mode_id),
    )


def _finalize_career_batting(
    batting: pd.DataFrame, team_latest: pd.DataFrame
) -> pd.DataFrame:
    cols = [
        "player",
        "player_id",
        "team_latest",
        "matches",
        "innings",
        "runs_total",
        "balls_faced",
        "fours",
        "sixes",
        "dismissals",
        "average",
        "strike_rate",
        "not_outs",
        "ducks",
        "fifties",
        "seventyfives",
        "hundreds",
        "fantasy_batting_total",
        "fantasy_batting_avg",
        "fantasy_batting_max",
        "fantasy_batting_min",
    ]
    out = batting.merge(team_latest, on="player", how="left")
    out["team_latest"] = out["team_latest"].fillna("UNKNOWN")
    return out.loc[:, cols].sort_values("player").reset_index(drop=True)


def _finalize_career_bowling(
    bowling: pd.DataFrame, team_latest: pd.DataFrame
) -> pd.DataFrame:
    cols = [
        "player",
        "player_id",
        "team_latest",
        "matches",
        "innings_bowled",
        "legal_deliveries",
        "overs_bowled",
        "runs_conceded",
        "wickets",
        "economy",
        "bowling_average",
        "maidens",
        "four_wicket_hauls",
        "fantasy_bowling_total",
        "fantasy_bowling_avg",
        "fantasy_bowling_max",
        "fantasy_bowling_min",
    ]
    out = bowling.merge(team_latest, on="player", how="left")
    out["team_latest"] = out["team_latest"].fillna("UNKNOWN")
    return out.loc[:, cols].sort_values("player").reset_index(drop=True)


def _finalize_career_fielding(
    fielding: pd.DataFrame, team_latest: pd.DataFrame
) -> pd.DataFrame:
    cols = [
        "player",
        "player_id",
        "team_latest",
        "matches",
        "catches",
        "runouts",
        "stumpings",
        "fantasy_fielding_total",
        "fantasy_fielding_avg",
    ]
    out = fielding.merge(team_latest, on="player", how="left")
    out["team_latest"] = out["team_latest"].fillna("UNKNOWN")
    return out.loc[:, cols].sort_values("player").reset_index(drop=True)


def _finalize_venue_splits(
    venue_bat: pd.DataFrame, venue_bowl: pd.DataFrame, points: pd.DataFrame
) -> pd.DataFrame:
    vb = venue_bat.groupby(["player", "venue"], as_index=False).agg(
        matches=("match_id", "nunique"),
        innings=("match_id", "count"),
        runs_total=("runs_total", "sum"),
        balls_faced=("balls_faced", "sum"),
        dismissals=("dismissals", "sum"),
    )
    vb["strike_rate"] = vb.apply(
        lambda r: _safe_div(r["runs_total"] * 100.0, r["balls_faced"]), axis=1
    )
    vb["average"] = vb.apply(
        lambda r: _safe_div(r["runs_total"], r["dismissals"]), axis=1
    )

    vbowl = venue_bowl.groupby(["player", "venue"], as_index=False).agg(
        innings_bowled=("match_id", "nunique"),
        wickets=("wickets", "sum"),
        legal_deliveries=("legal_deliveries", "sum"),
        runs_conceded=("runs_conceded", "sum"),
    )
    vbowl["overs_bowled"] = vbowl["legal_deliveries"] / 6.0
    vbowl["economy"] = vbowl.apply(
        lambda r: _safe_div(r["runs_conceded"], r["overs_bowled"]), axis=1
    )

    fp = points.groupby(["player", "venue"], as_index=False).agg(
        fantasy_batting_avg=("fantasy_batting", "mean"),
        fantasy_bowling_avg=("fantasy_bowling", "mean"),
        fantasy_fielding_avg=("fantasy_fielding", "mean"),
        fantasy_total_avg=("fantasy_total", "mean"),
    )

    out = vb.merge(vbowl, on=["player", "venue"], how="outer").merge(
        fp, on=["player", "venue"], how="left"
    )
    out = out.fillna(
        {
            "matches": 0,
            "innings": 0,
            "runs_total": 0,
            "balls_faced": 0,
            "dismissals": 0,
            "strike_rate": 0.0,
            "average": 0.0,
            "innings_bowled": 0,
            "wickets": 0,
            "legal_deliveries": 0,
            "runs_conceded": 0,
            "overs_bowled": 0.0,
            "economy": 0.0,
            "fantasy_batting_avg": 0.0,
            "fantasy_bowling_avg": 0.0,
            "fantasy_fielding_avg": 0.0,
            "fantasy_total_avg": 0.0,
        }
    )
    cols = [
        "player",
        "venue",
        "matches",
        "innings",
        "runs_total",
        "balls_faced",
        "strike_rate",
        "average",
        "innings_bowled",
        "wickets",
        "overs_bowled",
        "economy",
        "fantasy_batting_avg",
        "fantasy_bowling_avg",
        "fantasy_fielding_avg",
        "fantasy_total_avg",
    ]
    return out.loc[:, cols].sort_values(["player", "venue"]).reset_index(drop=True)


def _finalize_phase_splits(
    phase_bat: pd.DataFrame, phase_bowl: pd.DataFrame, points: pd.DataFrame
) -> pd.DataFrame:
    bat = phase_bat.groupby(["player", "phase"], as_index=False).agg(
        innings_batted=("match_id", "nunique"),
        runs_total=("runs_total", "sum"),
        balls_faced=("balls_faced", "sum"),
        dot_balls=("dot_balls", "sum"),
    )
    bat["strike_rate"] = bat.apply(
        lambda r: _safe_div(r["runs_total"] * 100.0, r["balls_faced"]), axis=1
    )
    bat["dot_ball_pct"] = bat.apply(
        lambda r: _safe_div(r["dot_balls"] * 100.0, r["balls_faced"]), axis=1
    )

    bowl = phase_bowl.groupby(["player", "phase"], as_index=False).agg(
        innings_bowled=("match_id", "nunique"),
        legal_deliveries=("legal_deliveries", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        wickets=("wickets", "sum"),
    )
    bowl["economy"] = bowl.apply(
        lambda r: _safe_div(r["runs_conceded"], r["legal_deliveries"] / 6.0), axis=1
    )

    participation = pd.concat(
        [
            phase_bat.loc[:, ["match_id", "player", "phase"]],
            phase_bowl.loc[:, ["match_id", "player", "phase"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    fantasy_phase = participation.merge(
        points.loc[:, ["match_id", "player", "fantasy_batting", "fantasy_bowling"]],
        on=["match_id", "player"],
        how="left",
    )
    fantasy_phase = fantasy_phase.groupby(["player", "phase"], as_index=False).agg(
        fantasy_batting_avg=("fantasy_batting", "mean"),
        fantasy_bowling_avg=("fantasy_bowling", "mean"),
    )

    out = bat.merge(bowl, on=["player", "phase"], how="outer").merge(
        fantasy_phase.loc[
            :, ["player", "phase", "fantasy_batting_avg", "fantasy_bowling_avg"]
        ],
        on=["player", "phase"],
        how="left",
    )
    out = out.fillna(
        {
            "innings_batted": 0,
            "runs_total": 0,
            "balls_faced": 0,
            "strike_rate": 0.0,
            "dot_ball_pct": 0.0,
            "innings_bowled": 0,
            "legal_deliveries": 0,
            "runs_conceded": 0,
            "economy": 0.0,
            "wickets": 0,
            "fantasy_batting_avg": 0.0,
            "fantasy_bowling_avg": 0.0,
        }
    )
    cols = [
        "player",
        "phase",
        "innings_batted",
        "runs_total",
        "balls_faced",
        "strike_rate",
        "dot_ball_pct",
        "innings_bowled",
        "legal_deliveries",
        "runs_conceded",
        "economy",
        "wickets",
        "fantasy_batting_avg",
        "fantasy_bowling_avg",
    ]
    phase_order = {"powerplay": 0, "middle": 1, "death": 2}
    out["_p"] = out["phase"].map(phase_order).fillna(99)
    out = out.sort_values(["player", "_p"]).drop(columns=["_p"]).reset_index(drop=True)
    return out.loc[:, cols]


def _finalize_season_trends(
    season_bat: pd.DataFrame, season_bowl: pd.DataFrame, points: pd.DataFrame
) -> pd.DataFrame:
    bat = season_bat.groupby(["player", "season"], as_index=False).agg(
        runs_total=("runs_total", "sum"),
        balls_faced=("balls_faced", "sum"),
        dismissals=("dismissals", "sum"),
    )
    bat["strike_rate"] = bat.apply(
        lambda r: _safe_div(r["runs_total"] * 100.0, r["balls_faced"]), axis=1
    )
    bat["average"] = bat.apply(
        lambda r: _safe_div(r["runs_total"], r["dismissals"]), axis=1
    )

    bowl = season_bowl.groupby(["player", "season"], as_index=False).agg(
        wickets=("wickets", "sum"),
        legal_deliveries=("legal_deliveries", "sum"),
        runs_conceded=("runs_conceded", "sum"),
    )
    bowl["economy"] = bowl.apply(
        lambda r: _safe_div(r["runs_conceded"], r["legal_deliveries"] / 6.0), axis=1
    )

    fp = points.groupby(["player", "season"], as_index=False).agg(
        matches=("match_id", "nunique"),
        fantasy_total_avg=("fantasy_total", "mean"),
        fantasy_total_sum=("fantasy_total", "sum"),
    )
    out = fp.merge(bat, on=["player", "season"], how="left").merge(
        bowl, on=["player", "season"], how="left"
    )
    out = out.fillna(
        {
            "runs_total": 0,
            "strike_rate": 0.0,
            "average": 0.0,
            "wickets": 0,
            "economy": 0.0,
            "fantasy_total_avg": 0.0,
            "fantasy_total_sum": 0.0,
        }
    )
    cols = [
        "player",
        "season",
        "matches",
        "runs_total",
        "strike_rate",
        "average",
        "wickets",
        "economy",
        "fantasy_total_avg",
        "fantasy_total_sum",
    ]
    return out.loc[:, cols].sort_values(["player", "season"]).reset_index(drop=True)


@dataclass
class _MatchAccum:
    points_rows: list[pd.DataFrame]
    bat_innings_rows: list[pd.DataFrame]
    bowl_innings_rows: list[pd.DataFrame]
    field_match_rows: list[pd.DataFrame]
    venue_bat_rows: list[pd.DataFrame]
    venue_bowl_rows: list[pd.DataFrame]
    phase_bat_rows: list[pd.DataFrame]
    phase_bowl_rows: list[pd.DataFrame]
    season_bat_rows: list[pd.DataFrame]
    season_bowl_rows: list[pd.DataFrame]
    total_matches: int


def _new_accum() -> _MatchAccum:
    return _MatchAccum(
        points_rows=[],
        bat_innings_rows=[],
        bowl_innings_rows=[],
        field_match_rows=[],
        venue_bat_rows=[],
        venue_bowl_rows=[],
        phase_bat_rows=[],
        phase_bowl_rows=[],
        season_bat_rows=[],
        season_bowl_rows=[],
        total_matches=0,
    )


def _append_match_artifacts(acc: _MatchAccum, match_df: pd.DataFrame) -> None:
    """Score one match and append split fragments into accumulators.

    Args:
        acc: Mutable accumulator for concatenation after the scan.
        match_df: Normalized per-match deliveries (non-empty).
    """

    pts = calculate_match_points(match_df)
    if not pts.empty:
        acc.points_rows.append(pts)
    acc.bat_innings_rows.append(_batting_innings_rows(match_df))
    acc.bowl_innings_rows.append(_bowling_innings_rows(match_df))
    acc.field_match_rows.append(_fielding_match_rows(match_df))
    vbat, vbowl = _venue_splits_delivery_metrics(match_df)
    acc.venue_bat_rows.append(vbat)
    acc.venue_bowl_rows.append(vbowl)
    pbat, pbowl = _phase_splits_delivery_metrics(match_df)
    acc.phase_bat_rows.append(pbat)
    acc.phase_bowl_rows.append(pbowl)
    sbat, sbowl = _season_delivery_metrics(match_df)
    acc.season_bat_rows.append(sbat)
    acc.season_bowl_rows.append(sbowl)
    acc.total_matches += 1


def _process_matches(match_files: list[Path], run_id: str) -> _MatchAccum:
    """Load and score each parquet; accumulate rows for final CSV merges."""

    total_files = len(match_files)
    milestones = _progress_milestone_indices(total_files)
    log = logging.getLogger(__name__)
    log.info(
        "[run=%s] aggregate_all: begin parquet scan (%d file(s))",
        run_id,
        total_files,
    )
    acc = _new_accum()
    for idx, path in enumerate(match_files, start=1):
        match_df = pd.read_parquet(path)
        if match_df.empty:
            log.debug(
                "[run=%s] aggregate_all: skip empty parquet path=%s",
                run_id,
                path.name,
            )
            continue
        match_df = _ensure_delivery_people_id_columns(match_df)
        match_df = _with_normalized_venues(match_df)
        if log.isEnabledFor(logging.DEBUG):
            meta = match_df.iloc[0]
            log.debug(
                "[run=%s] aggregate_all: match_id=%s deliveries=%d venue=%s path=%s",
                run_id,
                str(meta.get("match_id", path.stem)),
                int(match_df.shape[0]),
                str(meta.get("venue", "")),
                path.name,
            )
        _append_match_artifacts(acc, match_df)
        if idx % 100 == 0:
            log.info(
                "[run=%s] aggregate_all: checkpoint %d/%d parquet files",
                run_id,
                idx,
                total_files,
            )
        if idx in milestones:
            pct = min(100, (idx * 100) // max(total_files, 1))
            log.info(
                "[run=%s] aggregate_all: progress %d/%d files (~%d%%)",
                run_id,
                idx,
                total_files,
                pct,
            )
    return acc


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@dataclass
class _ConcatedFrames:
    """All per-match fragments concatenated after a full parquet scan."""

    points: pd.DataFrame
    bat_innings: pd.DataFrame
    bowl_innings: pd.DataFrame
    field_rows: pd.DataFrame
    venue_bat: pd.DataFrame
    venue_bowl: pd.DataFrame
    phase_bat: pd.DataFrame
    phase_bowl: pd.DataFrame
    season_bat: pd.DataFrame
    season_bowl: pd.DataFrame


def _concat_match_accumulator(acc: _MatchAccum) -> _ConcatedFrames:
    """Flatten accumulator lists into ten working frames."""

    return _ConcatedFrames(
        points=_concat_or_empty(acc.points_rows),
        bat_innings=_concat_or_empty(acc.bat_innings_rows),
        bowl_innings=_concat_or_empty(acc.bowl_innings_rows),
        field_rows=_concat_or_empty(acc.field_match_rows),
        venue_bat=_concat_or_empty(acc.venue_bat_rows),
        venue_bowl=_concat_or_empty(acc.venue_bowl_rows),
        phase_bat=_concat_or_empty(acc.phase_bat_rows),
        phase_bowl=_concat_or_empty(acc.phase_bowl_rows),
        season_bat=_concat_or_empty(acc.season_bat_rows),
        season_bowl=_concat_or_empty(acc.season_bowl_rows),
    )


def _career_output_tables(c: _ConcatedFrames) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build career batting, bowling, and fielding tables."""

    team_latest = (
        _derive_team_latest(c.points)
        if not c.points.empty
        else pd.DataFrame(columns=["player", "team_latest"])
    )
    if not c.points.empty:
        batting = _finalize_career_batting(
            _add_fantasy_batting(_batting_career_from_innings(c.bat_innings), c.points),
            team_latest,
        )
        bowling = _finalize_career_bowling(
            _add_fantasy_bowling(_bowling_career_from_innings(c.bowl_innings), c.points),
            team_latest,
        )
        fielding = _finalize_career_fielding(
            _add_fantasy_fielding(_career_fielding_from_rows(c.field_rows), c.points),
            team_latest,
        )
        return batting, bowling, fielding
    batting = _finalize_career_batting(
        _batting_career_from_innings(c.bat_innings), team_latest
    )
    bowling = _finalize_career_bowling(
        _bowling_career_from_innings(c.bowl_innings), team_latest
    )
    fielding = _finalize_career_fielding(
        _career_fielding_from_rows(c.field_rows), team_latest
    )
    return batting, bowling, fielding


def _venue_phase_season_tables(c: _ConcatedFrames) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build venue, phase, and season split tables."""

    if c.points.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    venue = _finalize_venue_splits(c.venue_bat, c.venue_bowl, c.points)
    phase = _finalize_phase_splits(c.phase_bat, c.phase_bowl, c.points)
    season = _finalize_season_trends(c.season_bat, c.season_bowl, c.points)
    return venue, phase, season


def _outputs_from_concat(c: _ConcatedFrames) -> dict[str, pd.DataFrame]:
    """Produce the six CSV payloads from concatenated fragments."""

    cb, cbl, cf = _career_output_tables(c)
    vs, ps, st = _venue_phase_season_tables(c)
    return {
        "career_batting": cb,
        "career_bowling": cbl,
        "career_fielding": cf,
        "venue_splits": vs,
        "phase_splits": ps,
        "season_trends": st,
    }


def _players_active_in_latest_season(points: pd.DataFrame) -> tuple[set[str], int | None]:
    """Players with at least one scored match row in the dataset's latest season.

    Args:
        points: Concatenated ``calculate_match_points`` output (non-empty expected).

    Returns:
        (player_names, latest_season) or (empty set, None) if ``points`` is empty.
    """

    if points.empty or "season" not in points.columns or "player" not in points.columns:
        return set(), None
    latest = int(points["season"].max())
    active = set(points.loc[points["season"] == latest, "player"].astype(str).tolist())
    return active, latest


def _filter_outputs_to_players(
    outputs: dict[str, pd.DataFrame], players: set[str]
) -> dict[str, pd.DataFrame]:
    """Keep only rows whose ``player`` is in ``players`` for tables that have that column.

    Args:
        outputs: Six aggregate tables keyed by output name.
        players: Names to retain (typically latest-season actives).

    Returns:
        New dict of filtered DataFrames.
    """

    if not players:
        return outputs
    out: dict[str, pd.DataFrame] = {}
    for key, df in outputs.items():
        if df.empty or "player" not in df.columns:
            out[key] = df
        else:
            out[key] = df.loc[df["player"].isin(players)].reset_index(drop=True)
    return out


def _load_existing(paths: _AggPaths) -> dict[str, pd.DataFrame]:
    return {
        "career_batting": pd.read_csv(paths.career_batting),
        "career_bowling": pd.read_csv(paths.career_bowling),
        "career_fielding": pd.read_csv(paths.career_fielding),
        "venue_splits": pd.read_csv(paths.venue_splits),
        "phase_splits": pd.read_csv(paths.phase_splits),
        "season_trends": pd.read_csv(paths.season_trends),
    }


def _write_outputs(paths: _AggPaths, outputs: dict[str, pd.DataFrame]) -> None:
    outputs["career_batting"].to_csv(paths.career_batting, index=False)
    outputs["career_bowling"].to_csv(paths.career_bowling, index=False)
    outputs["career_fielding"].to_csv(paths.career_fielding, index=False)
    outputs["venue_splits"].to_csv(paths.venue_splits, index=False)
    outputs["phase_splits"].to_csv(paths.phase_splits, index=False)
    outputs["season_trends"].to_csv(paths.season_trends, index=False)


def _log_totals(points: pd.DataFrame, total_matches: int, run_id: str) -> None:
    """Emit completion summary for one aggregation run.

    Args:
        points: Concatenated per-player per-match fantasy rows (may be empty).
        total_matches: Number of non-empty parquet files processed.
        run_id: Correlation id for log lines.
    """

    log = logging.getLogger(__name__)
    if points.empty:
        log.info(
            "[run=%s] aggregate_all: done matches=%d (no fantasy rows)",
            run_id,
            total_matches,
        )
        return
    log.info(
        "[run=%s] aggregate_all: done matches=%d players=%d venues=%d seasons=%d→%d",
        run_id,
        total_matches,
        int(points["player"].nunique()),
        int(points["venue"].nunique()),
        int(points["season"].min()),
        int(points["season"].max()),
    )


def aggregate_all(
    processed_dir: str,
    output_dir: str,
    force: bool = False,
    active_latest_season_only: bool = True,
) -> dict[str, pd.DataFrame]:
    """Aggregate all per-match parquet files into analytical CSVs.

    Args:
        processed_dir: Directory of ``*.parquet`` match files, or a processed root
            that contains a ``matches/`` subfolder (as written by ``parse_all_matches``).
        output_dir: Path to output directory for aggregated CSVs.
        force: If True recompute even when output files already exist.
        active_latest_season_only: If True, drop players with no appearance in the
            latest calendar season present in the data (retired / inactive).

    Returns:
        Mapping of output CSV name to aggregated DataFrame.

    Note:
        When ``force=False`` and CSVs are loaded from disk, the filter is **not**
        re-applied; re-run with ``force=True`` to rebuild from parquets with a
        new ``active_latest_season_only`` value.
    """

    ensure_pipeline_logger(__name__)
    run_id = new_run_id()

    in_dir = Path(processed_dir)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    paths = _output_paths(out_dir)

    if (not force) and _all_outputs_exist(paths):
        logger.info(
            "[run=%s] aggregate_all: skip (outputs exist, force=False) dir=%s",
            run_id,
            str(out_dir),
        )
        return _load_existing(paths)

    logger.info(
        "[run=%s] aggregate_all: recompute force=%s active_latest_season_only=%s "
        "processed_dir=%s output_dir=%s",
        run_id,
        str(force),
        str(active_latest_season_only),
        str(in_dir),
        str(out_dir),
    )
    acc = _process_matches(_list_parquets(in_dir), run_id)
    concated = _concat_match_accumulator(acc)
    outputs = _outputs_from_concat(concated)
    if active_latest_season_only and not concated.points.empty:
        active, latest = _players_active_in_latest_season(concated.points)
        if active and latest is not None:
            before = len(outputs["career_batting"])
            outputs = _filter_outputs_to_players(outputs, active)
            logger.info(
                "[run=%s] aggregate_all: latest_season=%d active_players=%d "
                "career_batting_rows %d→%d",
                run_id,
                latest,
                len(active),
                before,
                len(outputs["career_batting"]),
            )
    _write_outputs(paths, outputs)
    logger.debug(
        "[run=%s] aggregate_all: wrote csv career_batting rows=%d career_bowling rows=%d "
        "career_fielding rows=%d venue_splits rows=%d phase_splits rows=%d season_trends rows=%d",
        run_id,
        len(outputs["career_batting"]),
        len(outputs["career_bowling"]),
        len(outputs["career_fielding"]),
        len(outputs["venue_splits"]),
        len(outputs["phase_splits"]),
        len(outputs["season_trends"]),
    )
    _log_totals(concated.points, acc.total_matches, run_id)
    return outputs

"""Four-way analysis for two franchises and a venue.

Runs: (1) squad venue splits, (2) venue batting vs bowling vs spin/pace profile,
(3) current-season fantasy averages from rosters JSON season, (4) batter–bowler
H2H between squads from per-match parquet deliveries.

Example:
    python -m src.scripts.team_venue_matchup_analysis --team-a MI --team-b PBKS \\
        --venue Wankhede --min-h2h-balls 12
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import ProjectPaths, get_project_paths, load_env

_REPO_ROOT = Path(__file__).resolve().parents[2]

_TEAM_SHORT_TO_FULL: Final[dict[str, str]] = {
    "CSK": "Chennai Super Kings",
    "DC": "Delhi Capitals",
    "GT": "Gujarat Titans",
    "KKR": "Kolkata Knight Riders",
    "LSG": "Lucknow Super Giants",
    "MI": "Mumbai Indians",
    "PBKS": "Punjab Kings",
    "RR": "Rajasthan Royals",
    "RCB": "Royal Challengers Bengaluru",
    "SRH": "Sunrisers Hyderabad",
}

_H2H_COLS: Final[list[str]] = [
    "batter",
    "bowler",
    "runs_batter",
    "is_legal_delivery",
    "is_wicket",
    "wicket_player_out",
    "is_bowler_wicket",
    "is_super_over",
]


class TeamVenueAnalysisRequest(BaseModel):
    """Validated CLI inputs for a two-team venue matchup report."""

    model_config = ConfigDict(str_strip_whitespace=True)

    team_a_short: str = Field(..., min_length=2, max_length=5)
    team_b_short: str = Field(..., min_length=2, max_length=5)
    venue_query: str = Field(..., min_length=2, max_length=120)
    min_h2h_balls: int = Field(default=6, ge=1, le=120)

    @field_validator("team_a_short", "team_b_short", mode="before")
    @classmethod
    def _upper_short(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("team short codes must be strings")
        return value.strip().upper()

    def resolved_team_names(self) -> tuple[str, str]:
        """Map short codes to franchise names from ``current_rosters`` keys."""

        a = _TEAM_SHORT_TO_FULL.get(self.team_a_short)
        b = _TEAM_SHORT_TO_FULL.get(self.team_b_short)
        if a is None:
            raise ValueError(f"Unknown team short code: {self.team_a_short!r}")
        if b is None:
            raise ValueError(f"Unknown team short code: {self.team_b_short!r}")
        return a, b


class TeamVenueScriptInvocation(BaseModel):
    """Full script invocation including optional output paths."""

    request: TeamVenueAnalysisRequest
    json_out: Path | None = None
    rosters_path: Path | None = None


@dataclass
class TeamVenueReportArtifacts:
    """Computed tables and metadata for stdout or JSON export."""

    team_a: str
    team_b: str
    venue: str
    season: int
    roster_file: Path
    manifest: Path
    v_players: pd.DataFrame
    profile: dict[str, float]
    season_tab: pd.DataFrame
    h2h_ab: pd.DataFrame
    h2h_ba: pd.DataFrame
    h2h_ab_f: pd.DataFrame
    h2h_ba_f: pd.DataFrame
    h2h_ab_batter_agg: pd.DataFrame
    h2h_ba_batter_agg: pd.DataFrame
    h2h_ab_bowler_agg: pd.DataFrame
    h2h_ba_bowler_agg: pd.DataFrame


def _safe_div(num: float, den: float) -> float:
    """Return num/den or 0.0 when den is zero."""

    return float(num / den) if den else 0.0


def _resolve_franchise_venue(query: str, venue_catalog: list[str]) -> str:
    """Pick a single canonical venue string from a user substring."""

    q = query.strip().lower()
    if not q:
        raise ValueError("venue query is empty")
    exact = [v for v in venue_catalog if v.lower() == q]
    if len(exact) == 1:
        return exact[0]
    starts = [v for v in venue_catalog if v.lower().startswith(q)]
    if len(starts) == 1:
        return starts[0]
    subs = sorted({v for v in venue_catalog if q in v.lower()})
    if not subs:
        raise ValueError(f"No venue contains or matches {query!r}")
    if len(subs) == 1:
        return subs[0]
    raise ValueError(f"Ambiguous venue {query!r}; candidates: {subs[:12]}")


def _load_roster_json(path: Path) -> dict[str, Any]:
    """Load roster reference JSON with teams and season."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read roster JSON path={path}: {exc}") from exc
    teams = raw.get("teams")
    if not isinstance(teams, dict):
        raise ValueError("roster JSON missing teams dict")
    return raw


def _players_for_team(teams: dict[str, Any], franchise: str) -> set[str]:
    """Return player display names for one franchise."""

    plist = teams.get(franchise)
    if not isinstance(plist, list):
        raise ValueError(f"Unknown franchise in roster: {franchise!r}")
    out: set[str] = set()
    for row in plist:
        if isinstance(row, dict) and row.get("name"):
            out.add(str(row["name"]))
    return out


def _bowler_type_map(teams: dict[str, Any]) -> dict[str, str]:
    """Map player name to spin | pace | other from roster role lists."""

    out: dict[str, str] = {}
    for plist in teams.values():
        if not isinstance(plist, list):
            continue
        for row in plist:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            roles = row.get("role", [])
            if not name or not isinstance(roles, list):
                continue
            role_strs = [str(x) for x in roles]
            if "Spin Bowler" in role_strs:
                out[str(name)] = "spin"
            elif "Pace Bowler" in role_strs:
                out[str(name)] = "pace"
            else:
                out.setdefault(str(name), "other")
    return out


def _write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Persist raw request parameters before derived work."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _venue_player_table(
    venue_splits: pd.DataFrame, venue: str, players: set[str]
) -> pd.DataFrame:
    """Rows from venue_splits for given players at one venue.

    View is batting + bowling only (no fielding).
    """

    sub = venue_splits.loc[
        (venue_splits["venue"] == venue) & (venue_splits["player"].isin(players))
    ].copy()
    rename = {
        "matches": "batting_matches",
        "innings": "batting_innings",
        "runs_total": "batting_runs_total",
        "balls_faced": "batting_balls_faced",
        "strike_rate": "batting_strike_rate",
        "average": "batting_average",
        "innings_bowled": "bowling_matches",
        "wickets": "bowling_wickets",
        "overs_bowled": "bowling_overs_bowled",
        "economy": "bowling_economy",
    }
    out = sub.rename(columns=rename).copy()
    out["matches_total"] = out[["batting_matches", "bowling_matches"]].max(axis=1)
    cols = [
        "player",
        "venue",
        "batting_matches",
        "batting_innings",
        "batting_runs_total",
        "batting_balls_faced",
        "batting_strike_rate",
        "batting_average",
        "fantasy_batting_avg",
        "bowling_matches",
        "bowling_wickets",
        "bowling_overs_bowled",
        "bowling_economy",
        "fantasy_bowling_avg",
        "matches_total",
    ]
    keep = [c for c in cols if c in out.columns]
    out = out.loc[:, keep].copy()
    if "fantasy_batting_avg" in out.columns and "fantasy_bowling_avg" in out.columns:
        out["fantasy_bat_plus_bowl_avg"] = (
            out["fantasy_batting_avg"].astype(float) + out["fantasy_bowling_avg"].astype(float)
        )
        out = out.sort_values(
            ["fantasy_bat_plus_bowl_avg", "fantasy_batting_avg"],
            ascending=[False, False],
        ).drop(columns=["fantasy_bat_plus_bowl_avg"])
    elif "fantasy_batting_avg" in out.columns:
        out = out.sort_values("fantasy_batting_avg", ascending=False)
    return out.reset_index(drop=True)


def _venue_league_profile(
    venue_splits: pd.DataFrame, venue: str, bowler_types: dict[str, str]
) -> dict[str, float]:
    """Innings-weighted fantasy signals at venue vs all venues."""

    wk = venue_splits.loc[venue_splits["venue"] == venue].copy()
    all_df = venue_splits

    def wmean(df: pd.DataFrame, val: str, w: str) -> float:
        m = df[val].astype(float) * df[w].astype(float)
        den = float(df[w].astype(float).sum())
        return float(m.sum() / den) if den else 0.0

    bat_wk = wk.loc[wk["innings"] >= 3]
    bowl_wk = wk.loc[wk["innings_bowled"] >= 3]
    bat_all = all_df.loc[all_df["innings"] >= 3]
    bowl_all = all_df.loc[all_df["innings_bowled"] >= 3]

    out: dict[str, float] = {
        "venue_bat_fantasy_wtd": wmean(bat_wk, "fantasy_batting_avg", "innings"),
        "allvenues_bat_fantasy_wtd": wmean(bat_all, "fantasy_batting_avg", "innings"),
        "venue_bowl_fantasy_wtd": wmean(bowl_wk, "fantasy_bowling_avg", "innings_bowled"),
        "allvenues_bowl_fantasy_wtd": wmean(
            bowl_all, "fantasy_bowling_avg", "innings_bowled"
        ),
    }
    bowl_wk = bowl_wk.assign(
        bt=bowl_wk["player"].map(bowler_types).fillna("unknown")
    )
    for label in ("spin", "pace"):
        s2 = bowl_wk.loc[bowl_wk["bt"] == label]
        out[f"venue_bowl_{label}_wtd"] = wmean(s2, "fantasy_bowling_avg", "innings_bowled")
        s3 = bowl_all.copy()
        s3["bt"] = s3["player"].map(bowler_types).fillna("unknown")
        s4 = s3.loc[s3["bt"] == label]
        out[f"allvenues_bowl_{label}_wtd"] = wmean(
            s4, "fantasy_bowling_avg", "innings_bowled"
        )
    return out


def _season_table(
    season_trends: pd.DataFrame, season: int, players: set[str]
) -> pd.DataFrame:
    """Per-player season row for current roster names.

    Sorted by ``fantasy_total_sum`` descending.
    """

    sub = season_trends.loc[
        (season_trends["season"] == season) & (season_trends["player"].isin(players))
    ].copy()
    return sub.sort_values("fantasy_total_sum", ascending=False).reset_index(drop=True)


def _agg_h2h_chunk(
    df: pd.DataFrame, batters: set[str], bowlers: set[str]
) -> pd.DataFrame:
    """Aggregate one parquet fragment for batter-in-batters vs bowler-in-bowlers."""

    d = df.loc[:, _H2H_COLS].copy()
    d = d.loc[~d["is_super_over"].fillna(False).astype(bool)]
    d = d[d["batter"].isin(batters) & d["bowler"].isin(bowlers)]
    if d.empty:
        return pd.DataFrame(
            columns=["batter", "bowler", "balls", "runs", "dismissals", "matches", "strike_rate"]
        )
    d["legal"] = d["is_legal_delivery"].fillna(False).astype(bool)
    d["out"] = (
        d["is_wicket"].fillna(False).astype(bool)
        & d["is_bowler_wicket"].fillna(False).astype(bool)
        & (d["wicket_player_out"].astype(str) == d["batter"].astype(str))
    )
    g = d.groupby(["batter", "bowler"], as_index=False).agg(
        balls=("legal", "sum"),
        runs=("runs_batter", "sum"),
        dismissals=("out", "sum"),
    )
    g["matches"] = 1
    g["strike_rate"] = (g["runs"] * 100.0 / g["balls"]).where(g["balls"] > 0, 0.0)
    return g


def _combine_h2h(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Sum balls/runs/dismissals across match files then recompute strike rate."""

    x = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if x.empty:
        return x
    g = (
        x.groupby(["batter", "bowler"], as_index=False)
        .agg(
            balls=("balls", "sum"),
            runs=("runs", "sum"),
            dismissals=("dismissals", "sum"),
            matches=("matches", "sum"),
        )
    )
    g["strike_rate"] = g.apply(lambda r: _safe_div(r["runs"] * 100.0, r["balls"]), axis=1)
    return g


def _h2h_batter_aggregate_vs_pool(h2h_df: pd.DataFrame, min_balls: int) -> pd.DataFrame:
    """Summarise each batter against the entire opponent bowling pool.

    A batter is included only if their **total** legal balls faced across all
    listed opposing bowlers is at least ``min_balls``. ``strike_rate`` is
    aggregate: ``total_runs * 100 / balls_faced``.

    Args:
        h2h_df: Full directional H2H (all batter–bowler pairs).
        min_balls: Minimum total balls faced for inclusion.

    Returns:
        Columns ``batter``, ``total_runs``, ``dismissals``, ``balls_faced``,
        ``strike_rate``; sorted by ``total_runs`` descending.
    """

    if min_balls < 1:
        raise ValueError("min_balls must be at least 1")
    cols = ["batter", "total_runs", "dismissals", "balls_faced", "strike_rate"]
    if h2h_df.empty:
        return pd.DataFrame(columns=cols)
    g = h2h_df.groupby("batter", as_index=False).agg(
        total_runs=("runs", "sum"),
        dismissals=("dismissals", "sum"),
        balls_faced=("balls", "sum"),
    )
    g["strike_rate"] = g.apply(
        lambda r: _safe_div(float(r["total_runs"]) * 100.0, float(r["balls_faced"])),
        axis=1,
    )
    g = g.loc[g["balls_faced"] >= float(min_balls)].copy()
    return g.sort_values("total_runs", ascending=False).reset_index(drop=True)


def _h2h_bowler_aggregate_vs_pool(h2h_df: pd.DataFrame, min_balls: int) -> pd.DataFrame:
    """Summarise each bowler against the entire opponent batting pool.

    A bowler is included only if their **total** legal balls bowled to those
    batters is at least ``min_balls``. ``economy`` is ``runs_conceded`` per
    six balls: ``runs_conceded / (balls_bowled / 6)``.

    Args:
        h2h_df: Full directional H2H (all batter–bowler pairs).
        min_balls: Minimum total balls bowled for inclusion.

    Returns:
        Columns ``bowler``, ``wickets``, ``runs_conceded``, ``balls_bowled``,
        ``overs``, ``economy``; sorted by ``wickets`` descending.
    """

    if min_balls < 1:
        raise ValueError("min_balls must be at least 1")
    cols = ["bowler", "wickets", "runs_conceded", "balls_bowled", "overs", "economy"]
    if h2h_df.empty:
        return pd.DataFrame(columns=cols)
    g = h2h_df.groupby("bowler", as_index=False).agg(
        wickets=("dismissals", "sum"),
        runs_conceded=("runs", "sum"),
        balls_bowled=("balls", "sum"),
    )
    g["overs"] = g["balls_bowled"] / 6.0
    g["economy"] = g.apply(
        lambda r: _safe_div(float(r["runs_conceded"]), float(r["balls_bowled"]) / 6.0),
        axis=1,
    )
    g = g.loc[g["balls_bowled"] >= float(min_balls)].copy()
    return g.sort_values(
        ["wickets", "runs_conceded"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def _h2h_rows_meeting_min_balls(df: pd.DataFrame, min_balls: int) -> pd.DataFrame:
    """H2H pairs where the batter faced at least ``min_balls`` legal deliveries.

    Rows are sorted by ``strike_rate`` ascending (stable sort). Adds
    ``bowler_effectiveness`` as ``dismissals / matches`` (dismissals per match
    in which this pair faced each other).

    Args:
        df: Aggregated H2H with ``balls``, ``matches``, ``dismissals``, ``strike_rate``.
        min_balls: Minimum legal balls faced for inclusion.

    Returns:
        Filtered and sorted DataFrame (may be empty).
    """

    if min_balls < 1:
        raise ValueError("min_balls must be at least 1")
    sub = df.loc[df["balls"] >= float(min_balls)].copy()
    m = sub["matches"].astype(float)
    sub["bowler_effectiveness"] = (
        (sub["dismissals"].astype(float) / m).where(m > 0, 0.0).astype(float)
    )
    return sub.sort_values(["batter", "strike_rate"], ascending=[True, True], kind="mergesort")


def _h2h_full_scan(
    matches_dir: Path, batters: set[str], bowlers: set[str]
) -> pd.DataFrame:
    """Scan all match parquets for one directional H2H (batters vs bowlers)."""

    parts: list[pd.DataFrame] = []
    for path in sorted(matches_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(path, columns=_H2H_COLS)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Parquet read failed path={path}: {exc}") from exc
        parts.append(_agg_h2h_chunk(df, batters, bowlers))
    return _combine_h2h(parts)


def _print_df(title: str, frame: pd.DataFrame) -> None:
    """Print a titled block or a placeholder when empty."""

    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")
    if frame.empty:
        print("(no rows)")
        return
    with pd.option_context("display.max_rows", 60, "display.width", 140):
        print(frame.to_string(index=False))


def _build_json_payload(
    req: TeamVenueAnalysisRequest,
    venue: str,
    team_a: str,
    team_b: str,
    season: int,
    v_players: pd.DataFrame,
    profile: dict[str, float],
    season_tab: pd.DataFrame,
    h2h_ab_filtered: pd.DataFrame,
    h2h_ba_filtered: pd.DataFrame,
    h2h_ab_batter_agg: pd.DataFrame,
    h2h_ba_batter_agg: pd.DataFrame,
    h2h_ab_bowler_agg: pd.DataFrame,
    h2h_ba_bowler_agg: pd.DataFrame,
) -> dict[str, Any]:
    """Serialize analysis outputs for optional JSON export.

    H2H pair tables must already satisfy per-pair ``req.min_h2h_balls``; pool
    batter and bowler aggregates use the same ``N`` on **total** balls faced /
    bowled respectively.
    """

    def df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

    return {
        "request": req.model_dump(),
        "resolved": {
            "team_a_full": team_a,
            "team_b_full": team_b,
            "venue": venue,
            "season": season,
            "h2h_min_balls_faced": req.min_h2h_balls,
        },
        "venue_player_splits": df_records(v_players),
        "venue_league_profile": profile,
        "season_trends": df_records(season_tab),
        "h2h_team_a_bat_vs_team_b_bowl": df_records(h2h_ab_filtered),
        "h2h_team_b_bat_vs_team_a_bowl": df_records(h2h_ba_filtered),
        "h2h_team_a_batters_vs_team_b_pool_aggregate": df_records(h2h_ab_batter_agg),
        "h2h_team_b_batters_vs_team_a_pool_aggregate": df_records(h2h_ba_batter_agg),
        "h2h_team_b_bowlers_vs_team_a_batting_pool_aggregate": df_records(
            h2h_ab_bowler_agg
        ),
        "h2h_team_a_bowlers_vs_team_b_batting_pool_aggregate": df_records(
            h2h_ba_bowler_agg
        ),
    }


def _parse_cli(argv: list[str] | None) -> TeamVenueScriptInvocation:
    """Parse argv into a validated ``TeamVenueScriptInvocation``."""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--team-a", required=True, help="Franchise short code, e.g. MI")
    p.add_argument("--team-b", required=True, help="Franchise short code, e.g. PBKS")
    p.add_argument("--venue", required=True, help="Venue substring or full name")
    p.add_argument(
        "--min-h2h-balls",
        "--min-balls",
        type=int,
        default=6,
        metavar="N",
        dest="min_h2h_balls",
        help=(
            "Minimum legal balls per H2H pair; same N for batter pool (balls faced) "
            "and bowler pool (balls bowled) aggregates (default: 6)"
        ),
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full report JSON",
    )
    p.add_argument(
        "--rosters",
        type=Path,
        default=None,
        help="Override path to current_rosters.json",
    )
    ns = p.parse_args(argv)
    req = TeamVenueAnalysisRequest(
        team_a_short=ns.team_a,
        team_b_short=ns.team_b,
        venue_query=ns.venue,
        min_h2h_balls=ns.min_h2h_balls,
    )
    return TeamVenueScriptInvocation(
        request=req, json_out=ns.json_out, rosters_path=ns.rosters
    )


def _emit_report(req: TeamVenueAnalysisRequest, art: TeamVenueReportArtifacts) -> None:
    """Print human-readable sections to stdout."""

    print(f"Resolved venue: {art.venue}")
    print(f"Roster season: {art.season} (from {art.roster_file})")
    print(f"Run manifest: {art.manifest}")
    t1, t2 = art.team_a, art.team_b
    _print_df(f"(1) Venue splits — {t1} & {t2} at {art.venue}", art.v_players)
    print(
        "\n(2) Venue profile (IPL-wide at this venue vs all venues, innings≥3 bat / "
        "innings_bowled≥3 bowl, roster-based spin|pace)"
    )
    for key in sorted(art.profile):
        print(f"  {key}: {art.profile[key]:.4f}")
    _print_df(f"(3) Season {art.season} trends — combined squads", art.season_tab)
    mb = req.min_h2h_balls
    print(
        f"\n(4) H2H — pairs: ≥{mb} legal balls **per batter–bowler**; "
        f"pool batter summaries: ≥{mb} balls **faced** total; "
        f"pool bowler summaries: ≥{mb} balls **bowled** total "
        f"(see --min-h2h-balls / --min-balls)."
    )
    _print_df(f"(4a) {t1} batters vs {t2} bowlers (by pair)", art.h2h_ab_f)
    _print_df(
        f"(4a-agg) {t1} batters vs {t2} pool — runs, outs, balls, SR (≥{mb} balls faced)",
        art.h2h_ab_batter_agg,
    )
    _print_df(
        f"(4a-bowl) {t2} bowlers vs {t1} pool — wickets, runs, economy (≥{mb} balls bowled; "
        f"sorted by wickets desc)",
        art.h2h_ab_bowler_agg,
    )
    _print_df(f"(4b) {t2} batters vs {t1} bowlers (by pair)", art.h2h_ba_f)
    _print_df(
        f"(4b-agg) {t2} batters vs {t1} pool — runs, outs, balls, SR (≥{mb} balls faced)",
        art.h2h_ba_batter_agg,
    )
    _print_df(
        f"(4b-bowl) {t1} bowlers vs {t2} pool — wickets, runs, economy (≥{mb} balls bowled; "
        f"sorted by wickets desc)",
        art.h2h_ba_bowler_agg,
    )


def _squads_from_roster(
    roster_file: Path, req: TeamVenueAnalysisRequest
) -> tuple[str, str, set[str], set[str], int, dict[str, Any]]:
    """Resolve franchise names, player sets, season, and raw teams dict."""

    raw = _load_roster_json(roster_file)
    team_a, team_b = req.resolved_team_names()
    teams = raw["teams"]
    if not isinstance(teams, dict):
        raise ValueError("roster teams must be a dict")
    players_a = _players_for_team(teams, team_a)
    players_b = _players_for_team(teams, team_b)
    season = int(raw.get("season", 0))
    if season <= 0:
        raise ValueError("roster JSON missing positive season")
    return team_a, team_b, players_a, players_b, season, teams


def _new_analysis_manifest(req: TeamVenueAnalysisRequest, roster_file: Path) -> Path:
    """Create a timestamped JSON manifest under ``data/analysis_runs/``."""

    data_dir = _REPO_ROOT / "data"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = data_dir / "analysis_runs" / f"team_venue_analysis_{stamp}.json"
    _write_run_manifest(
        manifest,
        {
            "team_a_short": req.team_a_short,
            "team_b_short": req.team_b_short,
            "venue_query": req.venue_query,
            "min_h2h_balls": req.min_h2h_balls,
            "roster_path": str(roster_file),
        },
    )
    return manifest


def _venue_season_slice(
    req: TeamVenueAnalysisRequest,
    players_a: set[str],
    players_b: set[str],
    season: int,
    teams: dict[str, Any],
) -> tuple[str, pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Resolve venue string and build venue-player, profile, and season tables."""

    data_dir = _REPO_ROOT / "data"
    venue_splits = pd.read_csv(data_dir / "raw_aggregated_df_venue_splits.csv")
    season_trends = pd.read_csv(data_dir / "raw_aggregated_df_season_trends.csv")
    catalog = sorted(venue_splits["venue"].dropna().astype(str).unique().tolist())
    venue = _resolve_franchise_venue(req.venue_query, catalog)
    bowl_types = _bowler_type_map(teams)
    union = players_a | players_b
    v_players = _venue_player_table(venue_splits, venue, union)
    profile = _venue_league_profile(venue_splits, venue, bowl_types)
    season_tab = _season_table(season_trends, season, union)
    return venue, v_players, profile, season_tab


def _aggregate_and_h2h_artifacts(
    req: TeamVenueAnalysisRequest,
    roster_file: Path,
    manifest: Path,
    team_a: str,
    team_b: str,
    players_a: set[str],
    players_b: set[str],
    season: int,
    teams: dict[str, Any],
    paths: ProjectPaths,
) -> TeamVenueReportArtifacts:
    """Read aggregates, season rows, parquet H2H; H2H sorted by SR ascending."""

    venue, v_players, profile, season_tab = _venue_season_slice(
        req, players_a, players_b, season, teams
    )
    h2h_ab = _h2h_full_scan(paths.matches_dir, players_a, players_b)
    h2h_ba = _h2h_full_scan(paths.matches_dir, players_b, players_a)
    mb = req.min_h2h_balls
    ab_f = _h2h_rows_meeting_min_balls(h2h_ab, mb)
    ba_f = _h2h_rows_meeting_min_balls(h2h_ba, mb)
    ab_agg = _h2h_batter_aggregate_vs_pool(h2h_ab, mb)
    ba_agg = _h2h_batter_aggregate_vs_pool(h2h_ba, mb)
    ab_bowl_agg = _h2h_bowler_aggregate_vs_pool(h2h_ab, mb)
    ba_bowl_agg = _h2h_bowler_aggregate_vs_pool(h2h_ba, mb)
    return TeamVenueReportArtifacts(
        team_a=team_a,
        team_b=team_b,
        venue=venue,
        season=season,
        roster_file=roster_file,
        manifest=manifest,
        v_players=v_players,
        profile=profile,
        season_tab=season_tab,
        h2h_ab=h2h_ab,
        h2h_ba=h2h_ba,
        h2h_ab_f=ab_f,
        h2h_ba_f=ba_f,
        h2h_ab_batter_agg=ab_agg,
        h2h_ba_batter_agg=ba_agg,
        h2h_ab_bowler_agg=ab_bowl_agg,
        h2h_ba_bowler_agg=ba_bowl_agg,
    )


def _compute_report_artifacts(
    req: TeamVenueAnalysisRequest,
    roster_file: Path,
    paths: ProjectPaths,
) -> TeamVenueReportArtifacts:
    """Load rosters and aggregates, scan matches for H2H, filter H2H tables."""

    team_a, team_b, pa, pb, season, teams = _squads_from_roster(roster_file, req)
    manifest = _new_analysis_manifest(req, roster_file)
    return _aggregate_and_h2h_artifacts(
        req, roster_file, manifest, team_a, team_b, pa, pb, season, teams, paths
    )


def main(argv: list[str] | None = None) -> int:
    """Load data, run four analyses, print report, optional JSON export."""

    load_env()
    paths = get_project_paths()
    inv = _parse_cli(argv if argv is not None else sys.argv[1:])
    req = inv.request
    if req.team_a_short == req.team_b_short:
        print("error: --team-a and --team-b must differ", file=sys.stderr)
        return 2
    roster_file = inv.rosters_path or (_REPO_ROOT / "data/reference/current_rosters.json")
    art = _compute_report_artifacts(req, roster_file, paths)
    _emit_report(req, art)
    if inv.json_out is not None:
        payload = _build_json_payload(
            req,
            art.venue,
            art.team_a,
            art.team_b,
            art.season,
            art.v_players,
            art.profile,
            art.season_tab,
            art.h2h_ab_f,
            art.h2h_ba_f,
            art.h2h_ab_batter_agg,
            art.h2h_ba_batter_agg,
            art.h2h_ab_bowler_agg,
            art.h2h_ba_bowler_agg,
        )
        inv.json_out.parent.mkdir(parents=True, exist_ok=True)
        inv.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {inv.json_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from e

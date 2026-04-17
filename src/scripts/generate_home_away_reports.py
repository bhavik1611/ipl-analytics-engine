"""Generate home/away static JSON reports for all team pairs.

This script uses:
  - Roster reference: ``data/reference/current_rosters.json``
  - Home venue map: ``data/reference/team_home_venues.json``
  - Aggregates: ``data/raw_aggregated_df_venue_splits.csv`` + ``data/raw_aggregated_df_season_trends.csv``
  - H2H ledger: ``data/processed/h2h_batter_bowler.parquet``

It produces one JSON per (home_team, away_team) at the home team's home venue.

Example:
    python -m src.scripts.generate_home_away_reports --min-h2h-balls 15
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]

TEAM_SHORT_TO_FULL: dict[str, str] = {
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
TEAM_FULL_TO_SHORT: dict[str, str] = {v: k for k, v in TEAM_SHORT_TO_FULL.items()}


@dataclass(frozen=True, slots=True)
class Inputs:
    roster_path: Path
    home_venues_path: Path
    venue_splits_csv: Path
    season_trends_csv: Path
    h2h_ledger_parquet: Path
    career_fielding_csv: Path
    fielding_venue_splits_csv: Path
    fielding_season_splits_csv: Path
    out_dir: Path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "../cric/web/public/analysis",
        help="Output directory for JSON files",
    )
    p.add_argument(
        "--min-h2h-balls",
        type=int,
        default=6,
        help="Minimum balls for H2H pair rows and pool aggregates",
    )
    p.add_argument(
        "--h2h-ledger",
        type=Path,
        default=_REPO_ROOT / "data/processed/h2h_batter_bowler.parquet",
        help="Precomputed H2H ledger parquet",
    )
    return p.parse_args(argv)


def _inputs_from_args(ns: argparse.Namespace) -> Inputs:
    return Inputs(
        roster_path=_REPO_ROOT / "data/reference/current_rosters.json",
        home_venues_path=_REPO_ROOT / "data/reference/team_home_venues.json",
        venue_splits_csv=_REPO_ROOT / "data/raw_aggregated_df_venue_splits.csv",
        season_trends_csv=_REPO_ROOT / "data/raw_aggregated_df_season_trends.csv",
        h2h_ledger_parquet=Path(ns.h2h_ledger),
        career_fielding_csv=_REPO_ROOT / "data/raw_aggregated_df_career_fielding.csv",
        fielding_venue_splits_csv=_REPO_ROOT / "data/processed/fielding_venue_splits.csv",
        fielding_season_splits_csv=_REPO_ROOT / "data/processed/fielding_season_splits.csv",
        out_dir=Path(ns.out_dir),
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _players_for_team(teams: dict[str, Any], franchise: str) -> set[str]:
    plist = teams.get(franchise)
    if not isinstance(plist, list):
        return set()
    return {str(r["name"]) for r in plist if isinstance(r, dict) and r.get("name")}


def _bowler_type_map(teams: dict[str, Any]) -> dict[str, str]:
    """Map player name to pace|spin|other using roster role tags."""

    out: dict[str, str] = {}
    for plist in teams.values():
        if not isinstance(plist, list):
            continue
        for row in plist:
            if not isinstance(row, dict) or not row.get("name"):
                continue
            name = str(row["name"])
            roles = row.get("role", [])
            role_strs = [str(x) for x in roles] if isinstance(roles, list) else []
            if "Spin Bowler" in role_strs:
                out[name] = "spin"
            elif "Pace Bowler" in role_strs:
                out[name] = "pace"
            else:
                out.setdefault(name, "other")
    return out


def _venue_profile(
    venue_splits: pd.DataFrame, venue: str, bowler_types: dict[str, str]
) -> dict[str, float]:
    wk = venue_splits.loc[venue_splits["venue"] == venue].copy()

    def wmean(df: pd.DataFrame, val: str, w: str) -> float:
        den = float(df[w].astype(float).sum())
        if den == 0:
            return 0.0
        return float((df[val].astype(float) * df[w].astype(float)).sum() / den)

    bat_wk = wk.loc[wk["innings"] >= 3]
    bowl_wk = wk.loc[wk["innings_bowled"] >= 3]
    bat_all = venue_splits.loc[venue_splits["innings"] >= 3]
    bowl_all = venue_splits.loc[venue_splits["innings_bowled"] >= 3]
    out = {
        "venue_bat_fantasy_wtd": wmean(bat_wk, "fantasy_batting_avg", "innings"),
        "allvenues_bat_fantasy_wtd": wmean(bat_all, "fantasy_batting_avg", "innings"),
        "venue_bowl_fantasy_wtd": wmean(bowl_wk, "fantasy_bowling_avg", "innings_bowled"),
        "allvenues_bowl_fantasy_wtd": wmean(bowl_all, "fantasy_bowling_avg", "innings_bowled"),
    }
    bowl_wk = bowl_wk.assign(bt=bowl_wk["player"].map(bowler_types).fillna("unknown"))
    bowl_all = bowl_all.assign(bt=bowl_all["player"].map(bowler_types).fillna("unknown"))
    for label in ("pace", "spin"):
        out[f"venue_bowl_{label}_wtd"] = wmean(
            bowl_wk.loc[bowl_wk["bt"] == label], "fantasy_bowling_avg", "innings_bowled"
        )
        out[f"allvenues_bowl_{label}_wtd"] = wmean(
            bowl_all.loc[bowl_all["bt"] == label], "fantasy_bowling_avg", "innings_bowled"
        )
    return out


def _h2h_pairs_filtered(h2h: pd.DataFrame, min_balls: int) -> pd.DataFrame:
    sub = h2h.loc[h2h["balls"] >= float(min_balls)].copy()
    return sub.sort_values(["batter", "strike_rate"], ascending=[True, True], kind="mergesort")


def _h2h_batter_agg(h2h: pd.DataFrame, min_balls: int) -> pd.DataFrame:
    cols = ["batter", "total_runs", "dismissals", "balls_faced", "strike_rate"]
    if h2h.empty:
        return pd.DataFrame(columns=cols)
    g = h2h.groupby("batter", as_index=False).agg(
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


def _h2h_bowler_agg(h2h: pd.DataFrame, min_balls: int) -> pd.DataFrame:
    cols = ["bowler", "wickets", "runs_conceded", "balls_bowled", "overs", "economy"]
    if h2h.empty:
        return pd.DataFrame(columns=cols)
    g = h2h.groupby("bowler", as_index=False).agg(
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
    return g.sort_values(["wickets", "runs_conceded"], ascending=[False, True], kind="mergesort").reset_index(
        drop=True
    )


def _fielding_career_overall(career_fielding: pd.DataFrame, players: set[str]) -> pd.DataFrame:
    """Overall career fielding rows for a player subset."""

    if career_fielding.empty or not players:
        return pd.DataFrame(
            columns=[
                "player",
                "matches",
                "catches",
                "runouts",
                "stumpings",
                "fantasy_fielding_total",
                "fantasy_fielding_avg",
            ]
        )
    cols = [
        "player",
        "matches",
        "catches",
        "runouts",
        "stumpings",
        "fantasy_fielding_total",
        "fantasy_fielding_avg",
    ]
    sub = career_fielding.loc[career_fielding["player"].isin(players), cols].copy()
    return sub.sort_values("fantasy_fielding_avg", ascending=False).reset_index(drop=True)


def _fielding_venue_wise(
    fielding_venue: pd.DataFrame, players: set[str], venue: str
) -> pd.DataFrame:
    """Fielding at a specific venue for a player subset."""

    cols = [
        "player",
        "venue",
        "matches",
        "catches",
        "runouts",
        "stumpings",
        "fantasy_fielding_total",
        "fantasy_fielding_avg",
    ]
    if fielding_venue.empty or not players:
        return pd.DataFrame(columns=cols)
    sub = fielding_venue.loc[
        (fielding_venue["venue"] == venue) & (fielding_venue["player"].isin(players)),
        cols,
    ].copy()
    return sub.sort_values("fantasy_fielding_avg", ascending=False).reset_index(drop=True)


def _fielding_season_wise(
    fielding_season: pd.DataFrame, players: set[str], season: int
) -> pd.DataFrame:
    """Fielding for a specific season for a player subset."""

    cols = [
        "player",
        "season",
        "matches",
        "catches",
        "runouts",
        "stumpings",
        "fantasy_fielding_total",
        "fantasy_fielding_avg",
    ]
    if fielding_season.empty or not players:
        return pd.DataFrame(columns=cols)
    sub = fielding_season.loc[
        (fielding_season["season"] == season) & (fielding_season["player"].isin(players)),
        cols,
    ].copy()
    return sub.sort_values("fantasy_fielding_total", ascending=False).reset_index(drop=True)


def _payload(
    *,
    home: str,
    away: str,
    venue: str,
    season: int,
    min_h2h_balls: int,
    venue_player_splits: pd.DataFrame,
    venue_league_profile: dict[str, float],
    season_trends: pd.DataFrame,
    h2h_ab: pd.DataFrame,
    h2h_ba: pd.DataFrame,
    h2h_bat_ab: pd.DataFrame,
    h2h_bat_ba: pd.DataFrame,
    h2h_bowl_ab: pd.DataFrame,
    h2h_bowl_ba: pd.DataFrame,
    fielding_career_overall: pd.DataFrame,
    fielding_venue_wise: pd.DataFrame,
    fielding_season_wise: pd.DataFrame,
) -> dict[str, Any]:
    def df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

    return {
        "request": {
            "team_a_short": TEAM_FULL_TO_SHORT[home],
            "team_b_short": TEAM_FULL_TO_SHORT[away],
            "venue_query": venue,
            "min_h2h_balls": min_h2h_balls,
        },
        "resolved": {
            "team_a_full": home,
            "team_b_full": away,
            "venue": venue,
            "season": season,
            "h2h_min_balls_faced": min_h2h_balls,
        },
        "venue_player_splits": df_records(venue_player_splits),
        "venue_league_profile": venue_league_profile,
        "season_trends": df_records(season_trends),
        "h2h_team_a_bat_vs_team_b_bowl": df_records(h2h_ab),
        "h2h_team_b_bat_vs_team_a_bowl": df_records(h2h_ba),
        "h2h_team_a_batters_vs_team_b_pool_aggregate": df_records(h2h_bat_ab),
        "h2h_team_b_batters_vs_team_a_pool_aggregate": df_records(h2h_bat_ba),
        "h2h_team_b_bowlers_vs_team_a_batting_pool_aggregate": df_records(h2h_bowl_ab),
        "h2h_team_a_bowlers_vs_team_b_batting_pool_aggregate": df_records(h2h_bowl_ba),
        "fielding_overall_career": df_records(fielding_career_overall),
        "fielding_venue_wise": df_records(fielding_venue_wise),
        "fielding_season_wise": df_records(fielding_season_wise),
    }


def _write_json(out_dir: Path, filename: str, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / filename
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _index_row(
    *, out_path: Path, home: str, away: str, venue: str, season: int, min_h2h_balls: int
) -> dict[str, Any]:
    return {
        "id": out_path.stem,
        "home_team": home,
        "away_team": away,
        "home_short": TEAM_FULL_TO_SHORT[home],
        "away_short": TEAM_FULL_TO_SHORT[away],
        "venue": venue,
        "season": season,
        "min_h2h_balls": min_h2h_balls,
        "path": out_path.name,
    }


def _iter_home_away(franchises: list[str]) -> list[tuple[str, str]]:
    return [(h, a) for h in franchises for a in franchises if h != a]


def _venue_player_splits_for_union(
    venue_splits: pd.DataFrame, venue: str, union: set[str]
) -> pd.DataFrame:
    sub = venue_splits.loc[
        (venue_splits["venue"] == venue) & (venue_splits["player"].isin(union))
    ].copy()
    # Venue split view for UI: batting + bowling only (exclude fielding + total fantasy).
    # Rename ambiguous columns so the UI doesn't show "0 matches, 21 wickets".
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
        "fantasy_batting_avg": "fantasy_batting_avg",
        "fantasy_bowling_avg": "fantasy_bowling_avg",
    }
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
    out = sub.rename(columns=rename).copy()
    out["matches_total"] = out[["batting_matches", "bowling_matches"]].max(axis=1)
    keep = [c for c in cols if c in out.columns]
    out = out.loc[:, keep].copy()
    # Sort by batting+bowling fantasy signal (fallback to batting avg).
    if "fantasy_bowling_avg" in out.columns and "fantasy_batting_avg" in out.columns:
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


def _season_trends_for_union(
    season_trends_all: pd.DataFrame, season: int, union: set[str]
) -> pd.DataFrame:
    sub = season_trends_all.loc[
        (season_trends_all["season"] == season) & (season_trends_all["player"].isin(union))
    ].copy()
    return sub.sort_values("fantasy_total_sum", ascending=False).reset_index(drop=True)


def _h2h_directional(ledger: pd.DataFrame, batters: set[str], bowlers: set[str]) -> pd.DataFrame:
    return ledger.loc[ledger["batter"].isin(batters) & ledger["bowler"].isin(bowlers)].copy()


def _build_one_report(
    *,
    home: str,
    away: str,
    venue: str,
    season: int,
    min_h2h_balls: int,
    teams: dict[str, Any],
    venue_splits: pd.DataFrame,
    season_trends_all: pd.DataFrame,
    ledger: pd.DataFrame,
    bowler_types: dict[str, str],
    career_fielding: pd.DataFrame,
    fielding_venue: pd.DataFrame,
    fielding_season: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    ph = _players_for_team(teams, home)
    pa = _players_for_team(teams, away)
    union = ph | pa
    vsub = _venue_player_splits_for_union(venue_splits, venue, union)
    profile = _venue_profile(venue_splits, venue, bowler_types)
    ssub = _season_trends_for_union(season_trends_all, season, union)
    ab = _h2h_directional(ledger, ph, pa)
    ba = _h2h_directional(ledger, pa, ph)
    field_overall = _fielding_career_overall(career_fielding, union)
    field_venue = _fielding_venue_wise(fielding_venue, union, venue)
    field_season = _fielding_season_wise(fielding_season, union, season)
    payload = _payload(
        home=home,
        away=away,
        venue=venue,
        season=season,
        min_h2h_balls=min_h2h_balls,
        venue_player_splits=vsub,
        venue_league_profile=profile,
        season_trends=ssub,
        h2h_ab=_h2h_pairs_filtered(ab, min_h2h_balls),
        h2h_ba=_h2h_pairs_filtered(ba, min_h2h_balls),
        h2h_bat_ab=_h2h_batter_agg(ab, min_h2h_balls),
        h2h_bat_ba=_h2h_batter_agg(ba, min_h2h_balls),
        h2h_bowl_ab=_h2h_bowler_agg(ab, min_h2h_balls),
        h2h_bowl_ba=_h2h_bowler_agg(ba, min_h2h_balls),
        fielding_career_overall=field_overall,
        fielding_venue_wise=field_venue,
        fielding_season_wise=field_season,
    )
    fname = f"{TEAM_FULL_TO_SHORT[home]}__{TEAM_FULL_TO_SHORT[away]}__{venue.replace(' ', '_')}.json"
    return fname, payload


def _pair_reports(
    franchises: list[str],
    teams: dict[str, Any],
    home_venues: dict[str, Any],
    venue_splits: pd.DataFrame,
    season_trends_all: pd.DataFrame,
    ledger: pd.DataFrame,
    season: int,
    min_h2h_balls: int,
    out_dir: Path,
    bowler_types: dict[str, str],
    career_fielding: pd.DataFrame,
    fielding_venue: pd.DataFrame,
    fielding_season: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = _iter_home_away(franchises)
    it = tqdm(pairs, total=len(pairs), desc="Generating home/away reports", unit="matchup")
    for home, away in it:
        venue = str(home_venues[home])
        fname, payload = _build_one_report(
            home=home,
            away=away,
            venue=venue,
            season=season,
            min_h2h_balls=min_h2h_balls,
            teams=teams,
            venue_splits=venue_splits,
            season_trends_all=season_trends_all,
            ledger=ledger,
            bowler_types=bowler_types,
            career_fielding=career_fielding,
            fielding_venue=fielding_venue,
            fielding_season=fielding_season,
        )
        out_path = _write_json(out_dir, fname, payload)
        rows.append(
            _index_row(
                out_path=out_path,
                home=home,
                away=away,
                venue=venue,
                season=season,
                min_h2h_balls=min_h2h_balls,
            )
        )
    return rows


def generate_all(inv: Inputs, min_h2h_balls: int) -> Path:
    if min_h2h_balls < 1:
        raise ValueError("min_h2h_balls must be at least 1")
    roster = _load_json(inv.roster_path)
    teams = roster.get("teams", {})
    season = int(roster.get("season", 0))
    home_venues = _load_json(inv.home_venues_path).get("teams", {})
    if not isinstance(teams, dict) or not isinstance(home_venues, dict) or season <= 0:
        raise ValueError("Invalid roster or home venue mapping")
    venue_splits = pd.read_csv(inv.venue_splits_csv)
    season_trends_all = pd.read_csv(inv.season_trends_csv)
    ledger = pd.read_parquet(inv.h2h_ledger_parquet)
    career_fielding = pd.read_csv(inv.career_fielding_csv)
    fielding_venue = pd.read_csv(inv.fielding_venue_splits_csv)
    fielding_season = pd.read_csv(inv.fielding_season_splits_csv)
    franchises = sorted(teams.keys())
    bowler_types = _bowler_type_map(teams)
    index_rows = _pair_reports(
        franchises,
        teams,
        home_venues,
        venue_splits,
        season_trends_all,
        ledger,
        season,
        min_h2h_balls,
        inv.out_dir,
        bowler_types,
        career_fielding,
        fielding_venue,
        fielding_season,
    )
    idx = inv.out_dir / "index.json"
    idx.write_text(json.dumps({"schema_version": 1, "reports": index_rows}, indent=2), encoding="utf-8")
    return idx


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    inv = _inputs_from_args(ns)
    idx = generate_all(inv, int(ns.min_h2h_balls))
    print(f"Wrote reports index: {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Fantasy scoring calculator (pure, stateless).

This module implements match-level fantasy points from a ball-by-ball
deliveries DataFrame. It performs no I/O and has no LangChain dependencies.
"""

from __future__ import annotations

from typing import Final

import pandas as pd


_RETIRED_HURT_KINDS: Final[set[str]] = {"retired hurt", "retired ill", "retd hurt"}


def _filter_super_overs(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out super-over deliveries.

    Invariants satisfied:
        1. Super over rows (is_super_over == True) excluded from ALL points

    Args:
        df: Match deliveries DataFrame (may include super overs).

    Returns:
        A filtered DataFrame excluding rows where is_super_over is True.
    """

    return df.loc[~df["is_super_over"]].copy()  # noqa: E712


def _is_maiden_over(over_deliveries: pd.DataFrame) -> bool:
    """Return True if the over is a maiden under fantasy rules.

    Invariants satisfied:
        7. Maiden: any is_wide or is_noball in the over disqualifies it

    Args:
        over_deliveries: All deliveries for a single over for a single bowler.

    Returns:
        True if all runs_batter == 0 and there are no wides or no-balls.
    """

    if over_deliveries.empty:
        return False
    legal_balls = int(over_deliveries.loc[over_deliveries["is_legal_delivery"]].shape[0])  # noqa: E712
    if legal_balls != 6:
        return False
    if not (over_deliveries["runs_batter"] == 0).all():
        return False
    if over_deliveries["is_wide"].any():
        return False
    if over_deliveries["is_noball"].any():
        return False
    return True


def _bowler_economy(innings_df: pd.DataFrame, bowler: str) -> tuple[float, float]:
    """Compute (runs_conceded, overs_bowled) for an innings and bowler.

    Args:
        innings_df: Deliveries for a single innings (super overs already filtered).
        bowler: Bowler name.

    Returns:
        (runs_conceded, overs_bowled) where overs_bowled is legal_deliveries/6.
    """

    bowler_df = innings_df.loc[innings_df["bowler"] == bowler]
    runs_conceded = float(bowler_df["runs_bowler"].sum())
    legal_balls = int(bowler_df.loc[bowler_df["is_legal_delivery"]].shape[0])  # noqa: E712
    overs_bowled = legal_balls / 6.0
    return runs_conceded, overs_bowled


def _get_player_team(match_df: pd.DataFrame, player: str) -> str:
    """Infer the player's team for this match.

    Args:
        match_df: Deliveries for a single match (super overs may already be filtered).
        player: Player name.

    Returns:
        Team name, preferring batting_team where batter == player, else bowling_team
        where bowler == player. Returns "UNKNOWN" if not found.
    """

    batter_rows = match_df.loc[match_df["batter"] == player, "batting_team"]
    if not batter_rows.empty:
        return str(batter_rows.iloc[0])

    bowler_rows = match_df.loc[match_df["bowler"] == player, "bowling_team"]
    if not bowler_rows.empty:
        return str(bowler_rows.iloc[0])

    return "UNKNOWN"


def _batting_points_for_innings(innings_df: pd.DataFrame, player: str) -> int:
    """Compute batting points for one player in one innings.

    Invariants satisfied:
        3. A player earns batting + bowling + fielding points independently
        4. Retired hurt: runs + milestones only; no duck; no not-out bonus
        5. Retired out: −10 on top of normal batting points
        6. Run-out batter −10 is separate from and stacks with batting runs

    Args:
        innings_df: Deliveries for a single innings (super overs already filtered).
        player: Batter name.

    Returns:
        Batting fantasy points for this innings.
    """

    batter_df = innings_df.loc[innings_df["batter"] == player]
    if batter_df.empty:
        return 0

    runs = int(batter_df["runs_batter"].sum())
    balls_faced = int(
        batter_df.loc[batter_df["is_legal_delivery"]].shape[0]  # noqa: E712
    )
    played = balls_faced >= 1 or not batter_df.empty

    dismissal_rows = innings_df.loc[
        innings_df["is_wicket"]
        & (
            (innings_df["wicket_player_out"] == player)
            | (innings_df["wicket2_player_out"] == player)
        )
    ]
    dismissed = not dismissal_rows.empty
    wicket_kind = None
    if dismissed:
        row = dismissal_rows.iloc[0]
        if row.get("wicket_player_out") == player and not pd.isna(row.get("wicket_kind")):
            wicket_kind = str(row.get("wicket_kind"))
        elif row.get("wicket2_player_out") == player and row.get("wicket2_kind") is not None:
            wicket_kind = str(row.get("wicket2_kind"))

    retired_hurt = wicket_kind in _RETIRED_HURT_KINDS
    retired_out = wicket_kind == "retired out"
    run_out = wicket_kind == "run out"

    points = runs

    if runs >= 50:
        points += 10
    if runs >= 75:
        points += 10
    if runs >= 100:
        points += 10

    if played and dismissed and runs == 0 and not retired_hurt:
        points -= 10
    if run_out:
        points -= 10
    if retired_out:
        points -= 10

    if played and (not dismissed) and (not retired_hurt) and (runs > 0 or balls_faced >= 1):
        points += 10

    return int(points)


def _bowling_points_for_innings(innings_df: pd.DataFrame, player: str) -> int:
    """Compute bowling points for one player in one innings.

    Invariants satisfied:
        1. Super over rows (is_super_over == True) excluded from ALL points
        7. Maiden: any is_wide or is_noball in the over disqualifies it
        8. Economy bonus evaluated per innings; super overs excluded entirely

    Args:
        innings_df: Deliveries for a single innings (super overs already filtered).
        player: Bowler name.

    Returns:
        Bowling fantasy points for this innings.
    """

    bowler_df = innings_df.loc[(innings_df["bowler"] == player) & innings_df["is_wicket"]]
    wicket_df = bowler_df.loc[bowler_df["is_bowler_wicket"]]

    points = 0
    wicket_kinds = wicket_df["wicket_kind"].fillna("")
    points += int((wicket_kinds == "bowled").sum()) * 30
    points += int((wicket_kinds == "lbw").sum()) * 30
    other_mask = (wicket_kinds != "bowled") & (wicket_kinds != "lbw")
    points += int(other_mask.sum()) * 20

    wicket_count = int(wicket_df.shape[0])
    if wicket_count >= 4:
        points += 20

    bowler_deliveries = innings_df.loc[innings_df["bowler"] == player]
    maiden_overs = 0
    for _, over_deliveries in bowler_deliveries.groupby("over_1indexed", sort=False):
        if _is_maiden_over(over_deliveries):
            maiden_overs += 1
    points += maiden_overs * 30

    runs_conceded, overs_bowled = _bowler_economy(innings_df, player)
    if runs_conceded >= 50:
        points -= 10
    if runs_conceded >= 60:
        points -= 10
    if runs_conceded >= 70:
        points -= 10

    if overs_bowled >= 3.0 and overs_bowled > 0:
        economy = runs_conceded / overs_bowled
        if economy < 6.0:
            points += 10

    return int(points)


def _fielding_points_for_match(match_df: pd.DataFrame, player: str) -> int:
    """Compute fielding points for one player across the full match.

    Invariants satisfied:
        2. Substitute fielders (fielder_is_sub == True) earn zero fielding points
        3. A player earns batting + bowling + fielding points independently

    Args:
        match_df: Match deliveries DataFrame (super overs already filtered).
        player: Fielder name.

    Returns:
        Fielding fantasy points across the match.
    """

    def _is_sub_flag(value: object) -> bool:
        if value is None:
            return False
        return bool(value)

    def _is_present_name(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and pd.isna(value):
            return False
        if isinstance(value, str) and value.strip() == "":
            return False
        return True

    def _run_out_points_for_row(
        f1: object, f2: object, f1_is_sub: object, f2_is_sub: object, who: str
    ) -> int:
        """Compute run-out points for one wicket row for a single player.

        Invariants satisfied:
            2. Substitute fielders (fielder_is_sub == True) earn zero fielding points

        Args:
            f1: Primary run-out fielder.
            f2: Secondary run-out fielder (may be missing).
            f1_is_sub: Substitute flag for primary fielder (None => False).
            f2_is_sub: Substitute flag for secondary fielder (None => False).
            who: Player name being scored.

        Returns:
            Points awarded to `who` for this run-out event.
        """

        f1_ok = _is_present_name(f1) and (not _is_sub_flag(f1_is_sub))
        f2_ok = _is_present_name(f2) and (not _is_sub_flag(f2_is_sub))

        if f1_ok and f2_ok:
            if who == f1 or who == f2:
                return 10
            return 0
        if f1_ok and who == f1:
            return 20
        if f2_ok and who == f2:
            return 20
        return 0

    points = 0

    wk = match_df.loc[match_df["is_wicket"]]
    if wk.empty:
        return 0

    catch_mask = wk["wicket_kind"].isin({"caught", "caught and bowled"})
    catches = wk.loc[catch_mask & (wk["wicket_fielder1"] == player)]
    if not catches.empty:
        non_sub = catches["fielder1_is_sub"].apply(lambda v: not _is_sub_flag(v))
        points += int(non_sub.sum()) * 10

    caught_bowled = wk.loc[wk["wicket_kind"] == "caught and bowled"]
    if not caught_bowled.empty:
        credited = caught_bowled["wicket_fielder1"].apply(_is_present_name)
        missing_fielder = caught_bowled.loc[~credited]
        if not missing_fielder.empty:
            points += int((missing_fielder["bowler"] == player).sum()) * 10

    stump_mask = wk["wicket_kind"] == "stumped"
    stumpings = wk.loc[stump_mask & (wk["wicket_fielder1"] == player)]
    if not stumpings.empty:
        non_sub = stumpings["fielder1_is_sub"].apply(lambda v: not _is_sub_flag(v))
        points += int(non_sub.sum()) * 10

    runout_mask = wk["wicket_kind"] == "run out"
    runouts = wk.loc[runout_mask, ["wicket_fielder1", "wicket_fielder2", "fielder1_is_sub", "fielder2_is_sub"]]
    if not runouts.empty:
        points += int(
            runouts.apply(
                lambda r: _run_out_points_for_row(
                    r["wicket_fielder1"],
                    r["wicket_fielder2"],
                    r["fielder1_is_sub"],
                    r["fielder2_is_sub"],
                    player,
                ),
                axis=1,
            ).sum()
        )

    runout2_mask = wk["wicket2_kind"] == "run out"
    runouts2 = wk.loc[
        runout2_mask,
        ["wicket2_fielder1", "wicket2_fielder2", "wicket2_fielder1_is_sub", "wicket2_fielder2_is_sub"],
    ]
    if not runouts2.empty:
        points += int(
            runouts2.apply(
                lambda r: _run_out_points_for_row(
                    r["wicket2_fielder1"],
                    r["wicket2_fielder2"],
                    r["wicket2_fielder1_is_sub"],
                    r["wicket2_fielder2_is_sub"],
                    player,
                ),
                axis=1,
            ).sum()
        )

    return int(points)


def calculate_match_points(match_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate fantasy points for all players in a single match.

    Invariants satisfied:
        1. Super over rows (is_super_over == True) excluded from ALL points
        2. Substitute fielders (fielder_is_sub == True) earn zero fielding points
        3. A player earns batting + bowling + fielding points independently
        8. Economy bonus evaluated per innings; super overs excluded entirely

    Args:
        match_df: Deliveries DataFrame for exactly one match_id (all innings).

    Returns:
        DataFrame with one row per involved player and columns:
        match_id, player, team, fantasy_batting, fantasy_bowling,
        fantasy_fielding, fantasy_total, season, venue, date, is_playoff
    """

    cols = [
        "match_id",
        "player",
        "team",
        "fantasy_batting",
        "fantasy_bowling",
        "fantasy_fielding",
        "fantasy_total",
        "season",
        "venue",
        "date",
        "is_playoff",
    ]
    df = _filter_super_overs(match_df)
    if df.empty:
        return pd.DataFrame(columns=cols)

    meta = df.iloc[0]
    match_id, season = str(meta["match_id"]), int(meta["season"])
    venue, date, is_playoff = str(meta["venue"]), meta["date"], bool(meta["is_playoff"])

    players = set(df["batter"].dropna().astype(str)) | set(df["bowler"].dropna().astype(str))
    players |= set(df.loc[df["wicket_fielder1"].notna(), "wicket_fielder1"].astype(str))
    players |= set(df.loc[df["wicket_fielder2"].notna(), "wicket_fielder2"].astype(str))
    players |= set(df.loc[df["wicket2_fielder1"].notna(), "wicket2_fielder1"].astype(str))
    players |= set(df.loc[df["wicket2_fielder2"].notna(), "wicket2_fielder2"].astype(str))
    innings = list(df.groupby("innings_num", sort=False))

    rows: list[dict[str, object]] = []
    for player in sorted(players):
        batting = sum(_batting_points_for_innings(g, player) for _, g in innings)
        bowling = sum(_bowling_points_for_innings(g, player) for _, g in innings)
        fielding = _fielding_points_for_match(df, player)
        rows.append(
            {
                "match_id": match_id,
                "player": player,
                "team": _get_player_team(df, player),
                "fantasy_batting": int(batting),
                "fantasy_bowling": int(bowling),
                "fantasy_fielding": int(fielding),
                "fantasy_total": int(batting + bowling + fielding),
                "season": season,
                "venue": venue,
                "date": date,
                "is_playoff": is_playoff,
            }
        )
    return pd.DataFrame(rows, columns=cols)

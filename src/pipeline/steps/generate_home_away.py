"""Pipeline step: generate all static home/away matchup JSON files."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.models import StepResult, StepTiming
from src.pipeline.persistence import artifact_for_file, list_files, utc_now
from src.scripts.generate_home_away_reports import Inputs, generate_all


def run_generate_home_away(
    *,
    data_dir: Path,
    processed_dir: Path,
    out_dir: Path,
    min_h2h_balls: int,
) -> StepResult:
    """Generate all (home, away) matchup JSONs and an index file.

    Args:
        data_dir: Repository `data/` directory containing reference JSON + exported aggregates.
        processed_dir: Repository processed directory containing H2H and fielding outputs.
        out_dir: Output directory for `home_away/*.json`.
        min_h2h_balls: Minimum balls for H2H pair and pool aggregates.

    Returns:
        StepResult with input/output artifacts.
    """

    started = utc_now()
    inv = Inputs(
        roster_path=data_dir / "reference/current_rosters.json",
        home_venues_path=data_dir / "reference/team_home_venues.json",
        venue_splits_csv=data_dir / "raw_aggregated_df_venue_splits.csv",
        season_trends_csv=data_dir / "raw_aggregated_df_season_trends.csv",
        h2h_ledger_parquet=processed_dir / "h2h_batter_bowler.parquet",
        career_fielding_csv=data_dir / "raw_aggregated_df_career_fielding.csv",
        fielding_venue_splits_csv=processed_dir / "fielding_venue_splits.csv",
        fielding_season_splits_csv=processed_dir / "fielding_season_splits.csv",
        out_dir=out_dir,
    )
    idx = generate_all(inv, int(min_h2h_balls))
    ended = utc_now()

    inputs = [
        inv.roster_path,
        inv.home_venues_path,
        inv.venue_splits_csv,
        inv.season_trends_csv,
        inv.h2h_ledger_parquet,
        inv.career_fielding_csv,
        inv.fielding_venue_splits_csv,
        inv.fielding_season_splits_csv,
    ]
    inp = [artifact_for_file(p) for p in inputs]
    out_files = list_files(out_dir, "*.json")
    out = [artifact_for_file(p) for p in out_files]
    return StepResult(
        name="generate_home_away",
        timing=StepTiming(started_at_utc=started, ended_at_utc=ended),
        inputs=inp,
        outputs=out,
        notes=[f"out_dir={out_dir}", f"index={idx.name}", f"min_h2h_balls={min_h2h_balls}"],
    )


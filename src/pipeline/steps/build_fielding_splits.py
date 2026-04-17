"""Pipeline step: compute fielding split tables used by static reports."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.models import StepResult, StepTiming
from src.pipeline.persistence import artifact_for_file, list_files, utc_now
from src.scripts.build_fielding_splits import build_splits


def run_build_fielding_splits(
    *,
    matches_dir: Path,
    venue_csv: Path,
    season_csv: Path,
) -> StepResult:
    """Build venue-wise and season-wise fielding split CSVs.

    Args:
        matches_dir: Directory containing per-match parquets.
        venue_csv: Output CSV path for venue splits.
        season_csv: Output CSV path for season splits.

    Returns:
        StepResult with output artifacts and timing.
    """

    started = utc_now()
    venue, season = build_splits(matches_dir)
    venue_csv.parent.mkdir(parents=True, exist_ok=True)
    season_csv.parent.mkdir(parents=True, exist_ok=True)
    venue.to_csv(venue_csv, index=False)
    season.to_csv(season_csv, index=False)
    ended = utc_now()

    inp = [artifact_for_file(p) for p in list_files(matches_dir, "*.parquet")]
    out = [artifact_for_file(venue_csv), artifact_for_file(season_csv)]
    return StepResult(
        name="build_fielding_splits",
        timing=StepTiming(started_at_utc=started, ended_at_utc=ended),
        inputs=inp,
        outputs=out,
        notes=[f"venue_csv={venue_csv}", f"season_csv={season_csv}"],
    )


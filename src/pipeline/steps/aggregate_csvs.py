"""Pipeline step: per-match parquet → analytical aggregate CSVs."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.models import StepResult, StepTiming
from src.pipeline.persistence import artifact_for_file, list_files, utc_now
from src.utils.aggregator import aggregate_all


def run_aggregate_csvs(
    *,
    matches_dir: Path,
    aggregated_dir: Path,
    force: bool,
    active_latest_season_only: bool,
) -> StepResult:
    """Aggregate match parquets into 6 CSV tables.

    Args:
        matches_dir: Directory containing per-match parquets.
        aggregated_dir: Output directory for aggregate CSV files.
        force: When True, recompute even if outputs exist.
        active_latest_season_only: When True, drop inactive players from outputs.

    Returns:
        StepResult with inputs/outputs and timing.
    """

    started = utc_now()
    _ = aggregate_all(
        processed_dir=str(matches_dir),
        output_dir=str(aggregated_dir),
        force=force,
        active_latest_season_only=active_latest_season_only,
    )
    ended = utc_now()

    inp = [artifact_for_file(p) for p in list_files(matches_dir, "*.parquet")]
    out = [artifact_for_file(p) for p in list_files(aggregated_dir, "*.csv")]
    return StepResult(
        name="aggregate_csvs",
        timing=StepTiming(started_at_utc=started, ended_at_utc=ended),
        inputs=inp,
        outputs=out,
        notes=[
            f"aggregated_dir={aggregated_dir}",
            f"force={force}",
            f"active_latest_season_only={active_latest_season_only}",
        ],
    )


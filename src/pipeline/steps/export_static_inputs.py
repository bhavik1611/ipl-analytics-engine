"""Pipeline step: export aggregate CSVs to the static-report input filenames.

The static report generator historically reads:
- ``data/raw_aggregated_df_venue_splits.csv``
- ``data/raw_aggregated_df_season_trends.csv``
- ``data/raw_aggregated_df_career_fielding.csv``

This step bridges the `aggregate_all` outputs into those expected filenames,
while keeping the authoritative tables under ``data/processed/aggregated``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline.models import StepResult, StepTiming
from src.pipeline.persistence import artifact_for_file, utc_now


def _write_df(src: Path, dst: Path) -> None:
    """Read a CSV then write it to destination with stable settings."""

    df = pd.read_csv(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


def run_export_static_inputs(*, aggregated_dir: Path, data_dir: Path) -> StepResult:
    """Export specific aggregate tables to ``data/raw_aggregated_df_*.csv`` inputs.

    Args:
        aggregated_dir: Directory containing aggregate CSV outputs.
        data_dir: Repository ``data/`` directory.

    Returns:
        StepResult with created/overwritten input CSVs.
    """

    started = utc_now()
    mapping = [
        (aggregated_dir / "venue_splits.csv", data_dir / "raw_aggregated_df_venue_splits.csv"),
        (aggregated_dir / "season_trends.csv", data_dir / "raw_aggregated_df_season_trends.csv"),
        (aggregated_dir / "career_fielding.csv", data_dir / "raw_aggregated_df_career_fielding.csv"),
    ]
    for src, dst in mapping:
        _write_df(src, dst)
    ended = utc_now()

    return StepResult(
        name="export_static_inputs",
        timing=StepTiming(started_at_utc=started, ended_at_utc=ended),
        inputs=[artifact_for_file(p[0]) for p in mapping],
        outputs=[artifact_for_file(p[1]) for p in mapping],
        notes=[f"aggregated_dir={aggregated_dir}"],
    )


"""Pipeline step: derive current roster reference from latest-season parquets."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.models import StepResult, StepTiming
from src.pipeline.persistence import artifact_for_file, list_files, utc_now
from src.scripts.build_current_rosters import build_current_rosters


def run_build_rosters(*, matches_dir: Path, csv_out: Path, json_out: Path) -> StepResult:
    """Build roster artifacts used by static reports.

    Args:
        matches_dir: Directory containing per-match parquets.
        csv_out: Output CSV path.
        json_out: Output JSON path.

    Returns:
        StepResult with output artifacts.
    """

    started = utc_now()
    _ = build_current_rosters(matches_dir=matches_dir, csv_out=csv_out, json_out=json_out)
    ended = utc_now()

    inp = [artifact_for_file(p) for p in list_files(matches_dir, "*.parquet")]
    out = [artifact_for_file(csv_out), artifact_for_file(json_out)]
    return StepResult(
        name="build_rosters",
        timing=StepTiming(started_at_utc=started, ended_at_utc=ended),
        inputs=inp,
        outputs=out,
        notes=[f"csv_out={csv_out}", f"json_out={json_out}"],
    )


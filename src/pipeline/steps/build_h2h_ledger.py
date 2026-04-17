"""Pipeline step: build global batter–bowler H2H ledger parquet."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.models import StepResult, StepTiming
from src.pipeline.persistence import artifact_for_file, list_files, utc_now
from src.scripts.build_h2h_ledger import build_ledger


def run_build_h2h_ledger(*, matches_dir: Path, out_path: Path, force: bool) -> StepResult:
    """Build or refresh the H2H ledger parquet.

    Args:
        matches_dir: Directory containing per-match parquet files.
        out_path: Ledger parquet output path.
        force: When True, rebuild even if the output exists.

    Returns:
        StepResult describing ledger artifact.
    """

    started = utc_now()
    if out_path.is_file() and not force:
        ended = utc_now()
        return StepResult(
            name="build_h2h_ledger",
            timing=StepTiming(started_at_utc=started, ended_at_utc=ended),
            inputs=[artifact_for_file(p) for p in list_files(matches_dir, "*.parquet")],
            outputs=[artifact_for_file(out_path)],
            notes=[f"skip_existing=true out_path={out_path}"],
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = build_ledger(matches_dir, checkpoint=120)
    ledger.to_parquet(out_path, index=False)
    ended = utc_now()

    inp = [artifact_for_file(p) for p in list_files(matches_dir, "*.parquet")]
    out = [artifact_for_file(out_path)]
    return StepResult(
        name="build_h2h_ledger",
        timing=StepTiming(started_at_utc=started, ended_at_utc=ended),
        inputs=inp,
        outputs=out,
        notes=[f"force={force}", "checkpoint=120"],
    )


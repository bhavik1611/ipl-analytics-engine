"""Pipeline step: raw Cricsheet JSON → per-match parquet files."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.models import StepResult, StepTiming
from src.pipeline.persistence import artifact_for_file, list_files, utc_now
from src.utils.parser import parse_all_matches


def _raw_json_artifacts(raw_dir: Path) -> list[Path]:
    """Return raw match JSON file paths (for traceability)."""

    return list_files(raw_dir, "*.json")


def run_parse_matches(*, raw_dir: Path, processed_dir: Path, force: bool) -> StepResult:
    """Parse all raw match JSONs into per-match parquet files.

    Args:
        raw_dir: Directory containing Cricsheet IPL ``*.json`` matches.
        processed_dir: Processed output root directory.
        force: When True, overwrite existing match parquet files.

    Returns:
        StepResult with input/output artifacts and timing.
    """

    started = utc_now()
    _ = parse_all_matches(str(raw_dir), str(processed_dir), force=force)
    ended = utc_now()

    matches_dir = processed_dir / "matches"
    parquet_files = list_files(matches_dir, "*.parquet")
    out = [artifact_for_file(p) for p in parquet_files]
    inp = [artifact_for_file(p) for p in _raw_json_artifacts(raw_dir)]
    return StepResult(
        name="parse_matches",
        timing=StepTiming(started_at_utc=started, ended_at_utc=ended),
        inputs=inp,
        outputs=out,
        notes=[f"matches_dir={matches_dir}", f"force={force}"],
    )

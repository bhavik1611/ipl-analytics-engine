"""End-to-end orchestration for generating static analysis JSON reports."""

from __future__ import annotations

import os
from pathlib import Path

from src.config import get_project_paths, load_env
from src.pipeline.models import RunConfig, RunManifest, StepResult
from src.pipeline.persistence import ensure_dir, utc_now, write_json
from src.pipeline.steps.aggregate_csvs import run_aggregate_csvs
from src.pipeline.steps.build_fielding_splits import run_build_fielding_splits
from src.pipeline.steps.build_h2h_ledger import run_build_h2h_ledger
from src.pipeline.steps.build_rosters import run_build_rosters
from src.pipeline.steps.export_static_inputs import run_export_static_inputs
from src.pipeline.steps.generate_home_away import run_generate_home_away
from src.pipeline.steps.parse_matches import run_parse_matches
from src.utils.logging_support import ensure_pipeline_logger, new_run_id


def _env_snapshot() -> dict[str, str]:
    """Return a non-secret subset of env vars used by the run."""

    allow = {
        "CRICSHEET_RAW_DIR",
        "PROCESSED_DIR",
        "LOG_LEVEL",
    }
    out: dict[str, str] = {}
    for key in sorted(allow):
        val = os.getenv(key)
        if isinstance(val, str) and val.strip():
            out[key] = val.strip()
    return out


def _write_manifest(path: Path, manifest: RunManifest) -> None:
    """Persist the current manifest state."""

    write_json(path, manifest.model_dump())


def _append_step(manifest: RunManifest, step: StepResult) -> RunManifest:
    """Return a new manifest with an appended step."""

    return manifest.model_copy(update={"steps": [*manifest.steps, step]})


def _run_dir(base: Path, run_id: str) -> Path:
    """Return the per-run output directory under data/runs."""

    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return base / f"{stamp}_{run_id}"


def _init_manifest(*, config: RunConfig, run_id: str) -> RunManifest:
    """Create a fresh RunManifest for the run."""

    return RunManifest(
        run_id=run_id,
        created_at_utc=utc_now(),
        config=config,
        env=_env_snapshot(),
        steps=[],
    )


def _build_steps(*, config: RunConfig) -> list[StepResult]:
    """Execute all pipeline stages and return step results."""

    data_dir = Path("data")
    matches_dir = config.processed_dir / "matches"
    aggregated_dir = config.processed_dir / "aggregated"
    return [
        run_parse_matches(raw_dir=config.raw_dir, processed_dir=config.processed_dir, force=config.force),
        run_aggregate_csvs(
            matches_dir=matches_dir,
            aggregated_dir=aggregated_dir,
            force=config.force,
            active_latest_season_only=config.active_latest_season_only,
        ),
        run_export_static_inputs(aggregated_dir=aggregated_dir, data_dir=data_dir),
        run_build_rosters(
            matches_dir=matches_dir,
            csv_out=data_dir / "reference/current_rosters.csv",
            json_out=data_dir / "reference/current_rosters.json",
        ),
        run_build_fielding_splits(
            matches_dir=matches_dir,
            venue_csv=config.processed_dir / "fielding_venue_splits.csv",
            season_csv=config.processed_dir / "fielding_season_splits.csv",
        ),
        run_build_h2h_ledger(
            matches_dir=matches_dir,
            out_path=config.processed_dir / "h2h_batter_bowler.parquet",
            force=config.force,
        ),
        run_generate_home_away(
            data_dir=data_dir,
            processed_dir=config.processed_dir,
            out_dir=config.static_reports_dir,
            min_h2h_balls=config.min_h2h_balls,
        ),
    ]


def _persist_steps(
    *, manifest: RunManifest, manifest_path: Path, steps: list[StepResult], run_id: str
) -> RunManifest:
    """Append steps to manifest and write after each step."""

    log = ensure_pipeline_logger(__name__)
    current = manifest
    for step in steps:
        current = _append_step(current, step)
        _write_manifest(manifest_path, current)
        log.info("[run=%s] step=%s done outputs=%d", run_id, step.name, len(step.outputs))
    return current


def build_run_config(*, force: bool, min_h2h_balls: int) -> RunConfig:
    """Build a validated RunConfig using env-backed defaults."""

    paths = get_project_paths()
    repo_data = Path("data")
    return RunConfig(
        raw_dir=paths.cricsheet_raw_dir,
        processed_dir=paths.processed_dir,
        runs_dir=repo_data / "runs",
        static_reports_dir=Path("../cric/web/public/analysis"),
        force=force,
        active_latest_season_only=True,
        min_h2h_balls=min_h2h_balls,
    )


def run_static_reports(*, config: RunConfig) -> Path:
    """Run the full pipeline and return the manifest path.

    Args:
        config: Validated run configuration.

    Returns:
        Path to the written manifest JSON.
    """

    log = ensure_pipeline_logger(__name__)
    run_id = new_run_id()
    run_dir = _run_dir(config.runs_dir, run_id)
    ensure_dir(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = _init_manifest(config=config, run_id=run_id)
    _write_manifest(manifest_path, manifest)

    log.info("[run=%s] begin static reports pipeline", run_id)
    steps = _build_steps(config=config)
    _ = _persist_steps(manifest=manifest, manifest_path=manifest_path, steps=steps, run_id=run_id)
    log.info("[run=%s] pipeline complete manifest=%s", run_id, manifest_path)
    return manifest_path


def bootstrap_env(*, dotenv_path: str | None) -> None:
    """Load `.env` and ensure minimal directory structure exists."""

    load_env(dotenv_path=dotenv_path)
    paths = get_project_paths()
    ensure_dir(paths.processed_dir)
    ensure_dir(paths.processed_dir / "matches")

# 2026-04-17 — Static reports orchestrator (change log)

## Why this change

The repository needed a **single entry point** to generate all static analysis JSON files
end-to-end (specifically the `data/static_reports/home_away/` set), with strict persistence
for traceability and local quality checks.

## What changed (file-by-file)

- **`main.py`**
  - Repurposed to orchestrate the full pipeline for static report generation.

- **`src/pipeline/`**
  - New orchestration layer that runs each pipeline stage and writes a per-run `manifest.json`
    under `data/runs/<timestamp>_<run_id>/`.
  - Includes Pydantic models to keep run metadata structured and explicit.

- **`src/pipeline/steps/*`**
  - Step wrappers for:
    - parsing raw Cricsheet JSON → match parquets
    - aggregating parquets → analytical CSVs
    - exporting required `data/raw_aggregated_df_*.csv` bridge inputs for the JSON report generator
    - building rosters, fielding splits, H2H ledger
    - generating all home/away JSON reports + `index.json`

- **`src/scripts/build_current_rosters.py`**
  - Added public functions (`build_current_rosters`, `list_parquet_paths`, etc.) so the pipeline
    can call roster generation without importing protected/private helpers.

- **`scripts/quality_check.py`**
  - New local-only quality gate runner: CodeSense (if configured), Ruff, Pytest.

- **Docs**
  - `docs/system_design.md`: comprehensive architecture and flow documentation.
  - `docs/quality.md`: how to run the local quality gate (and configure CodeSense).

- **`requirements.txt`**
  - Added `ruff` (used by the quality gate).
  - Removed unused dependencies (`requests`, `beautifulsoup4`, `jupyter`) after confirming there are no first-party imports.

## Operational procedure

- Generate static reports:
  - `python main.py --min-h2h-balls 15`
  - Add `--force` to recompute.

- Run local checks:
  - `python scripts/quality_check.py`

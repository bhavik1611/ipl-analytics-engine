# IPL Analytics Engine (Static Reports)

This repository generates **static home/away matchup analysis JSON files** under `data/static_reports/home_away/` (plus `index.json`) from local Cricsheet IPL match JSON data.

## Prerequisites

- **Python**: 3.11+
- A local checkout of **Cricsheet IPL match JSONs** in a folder (defaults to `./data/raw/cricsheet_ipl`)

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a `.env` (optional) using `.env.example`:

```bash
cp .env.example .env
```

Environment variables:

- **`CRICSHEET_RAW_DIR`**: path to directory containing `*.json` match files
- **`PROCESSED_DIR`**: output directory for processed artifacts (parquet/CSVs)
- **`STATIC_REPORTS_DIR`** (optional): where `home_away` JSON reports are written (default: `data/static_reports/home_away`)
- **`LOG_LEVEL`**: `INFO` (default) or `DEBUG`

## Run the pipeline (end-to-end)

From the repo root:

```bash
python main.py --min-h2h-balls 15
```

To recompute from scratch:

```bash
python main.py --min-h2h-balls 15 --force
```

## Outputs

- **Match parquets**: `data/processed/matches/*.parquet`
- **Aggregate tables**: `data/processed/aggregated/*.csv`
- **Derived artifacts**:
  - `data/processed/fielding_venue_splits.csv`
  - `data/processed/fielding_season_splits.csv`
  - `data/processed/h2h_batter_bowler.parquet`
  - `data/reference/current_rosters.json`
- **Static reports**: `data/static_reports/home_away/*.json` and `data/static_reports/home_away/index.json`
- **Traceability manifest**: `data/runs/<timestamp>_<run_id>/manifest.json`

## Local quality checks

Run lint + tests:

```bash
python scripts/quality_check.py
```

## System design

See `docs/system_design.md` for the architecture and end-to-end flow.

## GitHub Actions: generate + publish to Vercel app

This repo includes a workflow that:

- downloads Cricsheet IPL JSON (`ipl_json.zip`)
- runs `main.py` to generate `data/static_reports/home_away/`
- copies the generated files into the Vercel app repo at `web/public/analysis`
- commits and pushes to `bhavik1611/fantasy-cric-app` (which triggers Vercel)

Workflow file: `.github/workflows/generate_static_analysis_and_publish.yml`

### Required secret (private app repo)

The workflow checks out and pushes to `bhavik1611/fantasy-cric-app`. The default `GITHUB_TOKEN` in this repo **cannot** access another private repository.

Add a **Personal Access Token** to **ipl-analytics-engine** → Settings → Secrets and variables → Actions:

- **`FANTASY_CRIC_APP_REPO_TOKEN`**: PAT that can read and write `fantasy-cric-app` contents (enough to clone and push to `main`).
  - **Classic PAT**: scope **`repo`** (simplest for a single private repo you control).
  - **Fine-grained PAT**: repository access **only** `fantasy-cric-app`, permission **Contents: Read and write**.

Create the token under the GitHub account that should appear as the commit author on the app repo (often your user), or a machine account with access to that repo.

**Previously:** the workflow used `FANTASY_CRIC_APP_DEPLOY_KEY` (SSH deploy key). That still works only if the **public** half is added under **fantasy-cric-app** → Settings → Deploy keys (with **Allow write access**), and the private key is valid OpenSSH PEM. Many “cannot fetch private repo” failures are fixed more reliably by switching to the PAT above, which is what the workflow uses now.

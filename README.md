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

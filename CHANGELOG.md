# Changelog (agent reference)

## 2026-04-17 — Static reports orchestrator + traceable runs

### Summary (parser ids)

Add a single pipeline entrypoint (`main.py`) that generates all static matchup JSONs under
`data/static_reports/home_away/` end-to-end from raw Cricsheet JSON inputs, and persist a
per-run manifest with input/output hashes for traceability.

### Operator usage

- Static reports pipeline:
  - `python main.py --min-h2h-balls 15`
  - Add `--force` to recompute from scratch.

### Key code touchpoints

- `main.py`: repo entrypoint now orchestrates the pipeline.
- `src/pipeline/`: new modular orchestration + persistence + step wrappers.
- `scripts/quality_check.py`: local-only quality gate (CodeSense + ruff + pytest).
- `docs/system_design.md`: architecture, flow, and persistence documentation.

## 2026-04-15 — Cricsheet `registry.people` ids (`player_id`)

### Summary (historical; agent stack removed)

Thread Cricsheet people hex ids from `info.registry.people` through match parquets, aggregation CSVs, narration metadata, Chroma ingest, retriever filters, and catalog. Display names remain the primary human-facing key; **`player_id`** is the canonical join key in metadata and new delivery columns.

### Operator pipeline (after pulling this change)

1. **Re-parse** raw Cricsheet JSON so match parquets include id columns: run your usual parse entrypoint with `force=True` on `parse_all_matches` (or delete `data/processed/matches/*.parquet` then re-run).
2. **Re-aggregate** so `career_*.csv` gain `player_id`: `aggregate_all(..., force=True)`.
3. **Note**: the earlier narration / Chroma ingest path has since been removed from this repository.

### Code touchpoints

| Area | Change |
| ---- | ------ |
| `src/utils/parser.py` | Reads `info.registry.people`; adds `batter_id`, `non_striker_id`, `bowler_id`, wicket/fielder `*_id` columns; exports `PEOPLE_ID_DELIVERY_COLUMNS` for backfill. |
| `src/utils/aggregator.py` | Backfills missing id columns on read; adds `player_id` to career batting/bowling/fielding CSVs (mode of delivery ids per name). |
| `src/utils/narrator.py` | Chroma-safe metadata `player_id` (`unknown` if missing); prose stays name-only. |
| `src/embeddings/ingest.py` | Comment: future text splits must copy metadata including `player_id`. |
| `src/tools/retriever.py` | Optional `player_id` filter; each hit includes `METADATA_JSON` plus header line. |
| `src/tools/catalog.py` | Facet `players_with_ids`; model `CatalogPlayerIdRow`. |

### Venues

No Cricsheet venue id in the IPL JSON set used here — **venue strings remain canonical** for venue metadata; ids deferred.

### Tests

`tests/test_parser.py`, `tests/test_retriever.py`, `tests/test_catalog.py`, `tests/test_narrator.py`; fixtures `match_all_cases_final.json` (registry), `match_registry_officials_deliveries.json`.

### Security

`snyk code test ./src` (medium+): **0 issues** at time of change.

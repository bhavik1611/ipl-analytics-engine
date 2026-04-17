# Local quality checks

## Why this exists

This repository is intended to be **maintainable and reproducible**. Before running the
data pipeline (or committing changes), use the local quality gate to catch common issues.

## One-command quality gate

Run:

- `python scripts/quality_check.py`

This enforces:

- **CodeSense score >= 9/10** (only if configured; see below)
- **Ruff** linting
- **Pytest**

## CodeSense integration (local-only)

The CodeSense plugin is typically IDE-integrated. For automation, this repo expects you
to provide a command that prints a score in a parseable format.

Configure:

- `CODESENSE_CMD`: the command to run CodeSense (must print something like `9.2/10`)
- `CODESENSE_MIN_SCORE` (optional): defaults to `9`

Example:

```bash
export CODESENSE_CMD="codesense review --format text"
export CODESENSE_MIN_SCORE="9"
python scripts/quality_check.py
```

If your CodeSense installation does not expose a CLI, leave `CODESENSE_CMD` unset and
the gate will skip the CodeSense check (Ruff + Pytest still run).

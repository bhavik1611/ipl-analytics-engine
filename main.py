"""Repo entrypoint: generate all static analysis JSON files.

This script orchestrates the end-to-end pipeline required to produce the static
home/away matchup JSON files under ``data/static_reports/home_away/``.
"""

from __future__ import annotations

import argparse
from src.pipeline.orchestrator import bootstrap_env, build_run_config, run_static_reports


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI flags for the static-reports pipeline."""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dotenv", default=".env", help="Optional .env path")
    p.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    p.add_argument("--min-h2h-balls", type=int, default=15, help="H2H inclusion threshold (balls)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the end-to-end static report generation pipeline.

    Args:
        argv: Optional argv list (without program name).

    Returns:
        Exit code.
    """

    args = _parse_args(argv)
    bootstrap_env(dotenv_path=args.dotenv or None)
    config = build_run_config(force=bool(args.force), min_h2h_balls=int(args.min_h2h_balls))
    _ = run_static_reports(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

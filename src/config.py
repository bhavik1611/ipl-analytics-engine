"""Project configuration (env-driven, no hardcoded paths).

All config is read from environment variables (optionally from a `.env` file)
to keep the codebase portable across machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved project directories.

    Attributes:
        cricsheet_raw_dir: Directory containing raw Cricsheet IPL JSON files.
        processed_dir: Directory for processed artifacts (parquet/CSV).
        matches_dir: Directory for per-match parquet outputs.
    """

    cricsheet_raw_dir: Path
    processed_dir: Path
    matches_dir: Path


def load_env(dotenv_path: str | None = None) -> None:
    """Load environment variables from a `.env` file, if present.

    Args:
        dotenv_path: Optional explicit `.env` path. If None, uses default lookup.
    """
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return
    load_dotenv(override=False)


def get_project_paths() -> ProjectPaths:
    """Resolve project paths from env vars with safe defaults.

    Returns:
        ProjectPaths with directories resolved as pathlib.Path.
    """
    processed_dir = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
    cricsheet_raw_dir = Path(os.getenv("CRICSHEET_RAW_DIR", "./data/raw/cricsheet_ipl"))
    return ProjectPaths(
        cricsheet_raw_dir=cricsheet_raw_dir,
        processed_dir=processed_dir,
        matches_dir=processed_dir / "matches",
    )

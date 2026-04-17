"""Persistence helpers for pipeline traceability.

These helpers are intentionally small and explicit to keep the pipeline
auditable. They avoid storing structured data as ad-hoc dicts by returning
Pydantic models where appropriate.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline.models import FileArtifact


def utc_now() -> datetime:
    """Return current UTC timestamp."""

    return datetime.now(timezone.utc)


def ensure_dir(path: Path) -> None:
    """Create a directory tree if missing."""

    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    """Return hex sha256 digest for a file."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_for_file(path: Path) -> FileArtifact:
    """Build a FileArtifact for a file path."""

    stat = path.stat()
    return FileArtifact(path=str(path), sha256=sha256_file(path), size_bytes=int(stat.st_size))


def write_json(path: Path, payload: object) -> None:
    """Write JSON to disk with stable formatting.

    Args:
        path: Output file path.
        payload: JSON-serializable object.
    """

    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def list_files(glob_root: Path, pattern: str) -> list[Path]:
    """List files under glob_root matching pattern (sorted)."""

    return sorted(glob_root.glob(pattern))

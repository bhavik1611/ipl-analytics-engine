"""Pipeline models (Pydantic v2) for reproducible runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FileArtifact(BaseModel):
    """One persisted file artifact with integrity metadata."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(..., description="Repository-relative or absolute path.")
    sha256: str = Field(..., min_length=64, max_length=64)
    size_bytes: int = Field(..., ge=0)


class StepTiming(BaseModel):
    """Start/end timestamps for one step."""

    model_config = ConfigDict(frozen=True)

    started_at_utc: datetime
    ended_at_utc: datetime


class StepResult(BaseModel):
    """Result of one pipeline step with inputs/outputs and timing."""

    model_config = ConfigDict(frozen=True)

    name: str
    timing: StepTiming
    inputs: list[FileArtifact] = Field(default_factory=list)
    outputs: list[FileArtifact] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RunConfig(BaseModel):
    """Validated configuration for a pipeline run."""

    model_config = ConfigDict(str_strip_whitespace=True)

    raw_dir: Path
    processed_dir: Path
    runs_dir: Path
    static_reports_dir: Path
    force: bool = False
    active_latest_season_only: bool = True
    min_h2h_balls: int = Field(default=15, ge=1, le=120)


class RunManifest(BaseModel):
    """Run manifest persisted to disk for traceability."""

    model_config = ConfigDict(str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    run_id: str
    created_at_utc: datetime
    config: RunConfig
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Non-secret environment snapshot used by the run.",
    )
    steps: list[StepResult] = Field(default_factory=list)


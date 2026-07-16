from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


SourceName = Literal["local", "db", "all"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class StageResult:
    name: str
    rows_in: int = 0
    rows_out: int = 0
    rejected: int = 0
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None
    message: str = ""
    code: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    status: str
    source: SourceName
    fetch_new: bool
    rows_loaded: int
    rows_clean: int
    rows_model_ready: int
    rows_rejected: int
    output_paths: dict[str, Path] = field(default_factory=dict)
    stage_results: tuple[StageResult, ...] = ()
    errors: tuple[str, ...] = ()
    finished_at: datetime = field(default_factory=utc_now)

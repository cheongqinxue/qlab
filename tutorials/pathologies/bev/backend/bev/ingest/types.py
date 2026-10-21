"""Typed payloads produced by the W&B reader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunInfo:
    """Metadata from a W&B ``RunRecord``."""

    wandb_run_id: str
    display_name: str
    project: str
    entity: str


@dataclass
class HistoryEvent:
    """One ``HistoryRecord`` normalized for catalog ingest."""

    step: int
    global_step: int | None = None
    epoch: float | None = None
    logged_at: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

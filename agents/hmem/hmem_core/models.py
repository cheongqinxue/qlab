"""Data models for H-MEM indexing and QA (paper-aligned types).

Pydantic models mirror community ``hmem.types`` where useful, kept experiment-local
per qlab import wall.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryScope(StrEnum):
    """Temporal granularity for tree retrieval (paper §3.3)."""

    SHORT = "SHORT"
    LONG = "LONG"
    MIXED = "MIXED"


class MemoryFragment(BaseModel):
    """Original agent memory fragment (conversation turn or event)."""

    id: str
    text: str
    timestamp: datetime
    speaker: str | None = None


class GraphEdge(BaseModel):
    """Resolved entity–relation–entity triple from a fragment."""

    head: str
    relation: str
    tail: str


class EntityNode(BaseModel):
    """Knowledge graph entity (profile fields added in Phase 2)."""

    name: str
    entity_type: str | None = None
    profile: str | None = None


class MemoryEvent(BaseModel):
    """Leaf-level semantic event linked to a fragment."""

    event_id: str
    fragment_id: str
    text: str
    timestamp: datetime


class TreeNode(BaseModel):
    """Node in the temporal–semantic tree."""

    node_id: str
    level: int
    text: str
    start_time: datetime
    end_time: datetime
    fragment_id: str | None = None
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    is_summary: bool = False
    consolidation_count: int = 0


class SubQuery(BaseModel):
    """Planned sub-query with scope and optional temporal hint."""

    text: str
    scope: MemoryScope = MemoryScope.MIXED
    depends_on: list[int] = Field(default_factory=list)
    time_start: datetime | None = None
    time_end: datetime | None = None


class Evidence(BaseModel):
    """Retrieved memory snippet before synthesis."""

    text: str
    fragment_id: str | None = None
    event_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    consolidation_count: int = 0
    score: float = 0.0
    semantic: float = 0.0
    temporal: float = 0.0
    robustness: float = 0.0


class AskResponse(BaseModel):
    """QA result with evidence count for observability."""

    query: str
    answer: str
    evidence_count: int
    agents_mode: str

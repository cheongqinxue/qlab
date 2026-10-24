"""HTTP request and response schemas for H-MEM."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from hmem_core.models import AskResponse


class HybridCountsOut(BaseModel):
    """Hybrid index counts."""

    fragments: int
    leaves: int
    entities: int
    summary_nodes: int
    indexed: bool


class HealthOut(BaseModel):
    """GET /health payload."""

    status: str
    experiment: str
    agents_mode: str
    hybrid: HybridCountsOut


class FragmentIn(BaseModel):
    """POST /corpus/fragments body."""

    id: str
    text: str
    timestamp: datetime
    speaker: str | None = None


class FragmentBatchOut(BaseModel):
    """Result of ingesting one or more fragments."""

    fragments: int
    leaves_added: int


class IndexOut(BaseModel):
    """POST /index payload."""

    summary_nodes: int
    indexed: bool


class BootstrapOut(BaseModel):
    """Toy bootstrap summary."""

    source: str
    fragments: int
    entities: int
    summary_nodes: int
    extraction: str = "fixture"


class AskIn(BaseModel):
    """POST /ask body."""

    query: str = Field(min_length=1, max_length=8_000)


class AskOut(AskResponse):
    """POST /ask response."""

    pass


class ResetOut(BaseModel):
    """POST /reset payload."""

    status: str

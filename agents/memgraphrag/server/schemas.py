"""HTTP response schemas for MemGraphRAG."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agents_utils.memory.base_memory import BaseMemoryItem


class MemoryCountsOut(BaseModel):
    """Memory tier counts."""

    ontology: int
    facts: int
    passages: int


class HealthOut(BaseModel):
    """GET /health payload."""

    status: str
    experiment: str
    memory: MemoryCountsOut


class PassageOut(BaseModel):
    """POST /corpus/passages payload."""

    passage_id: str
    passages: int


class CorpusCountOut(BaseModel):
    """Toy corpus load summary."""

    source: str
    passages: int


class BootstrapOut(BaseModel):
    """Toy bootstrap summary."""

    model_config = ConfigDict(extra="allow")

    source: str
    extraction: str = Field(default="fixture")


class IndexOut(BaseModel):
    """POST /index payload."""

    model_config = ConfigDict(extra="allow")


class RetrieveOut(BaseModel):
    """POST /retrieve payload."""

    query: str
    hits: list[BaseMemoryItem]


class ResetOut(BaseModel):
    """POST /reset payload."""

    status: str

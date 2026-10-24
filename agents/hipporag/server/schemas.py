"""HTTP response schemas for HippoRAG."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hipporag_core.models import RetrievalHit


class HealthOut(BaseModel):
    """GET /health payload."""

    status: str
    experiment: str
    passages: int
    entities: int
    triples: int
    rel_edges: int
    synonym_edges: int


class IndexOut(BaseModel):
    """POST /index payload."""

    indexed_passages: int
    synonym_edges_added: int
    passages: int
    entities: int
    triples: int


class StatusOut(BaseModel):
    """Simple status responses."""

    status: str


class PassageOut(BaseModel):
    """POST /corpus/passages payload."""

    passage_id: str
    passages: int


class CorpusCountOut(BaseModel):
    """Toy corpus load summary."""

    source: str
    passages: int


class BootstrapOut(BaseModel):
    """Toy bootstrap summary (extra index fields allowed)."""

    model_config = ConfigDict(extra="allow")

    source: str
    openie: str = Field(default="fixture")


class RetrieveOut(BaseModel):
    """POST /retrieve payload."""

    query: str
    query_entities: list[str]
    seed_entities: list[str]
    hits: list[RetrievalHit]


class AskOut(BaseModel):
    """POST /ask payload."""

    query: str
    answer: str
    hits: list[RetrievalHit]

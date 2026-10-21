"""Data models for passages, triples, and retrieval results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Passage(BaseModel):
    """Indexed text chunk."""

    id: str = Field(description="Stable passage identifier.")
    text: str = Field(min_length=1, description="Passage body.")


class Triple(BaseModel):
    """OpenIE-style subject–predicate–object edge from a passage."""

    subject: str
    predicate: str
    object: str
    passage_id: str = Field(description="Source passage for this triple.")


class RetrievalHit(BaseModel):
    """One passage returned by PPR-based retrieval."""

    passage_id: str
    score: float
    text: str

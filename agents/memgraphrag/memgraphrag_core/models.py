"""Data models for three-tier global memory."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PassageRecord(BaseModel):
    """Original text preserved for evidence (M_pas)."""

    id: str
    text: str = Field(min_length=1)


class FactRecord(BaseModel):
    """Concrete extracted fact (M_fac)."""

    id: str
    text: str = Field(min_length=1)
    passage_ids: list[str] = Field(default_factory=list)
    schema_ids: list[str] = Field(
        default_factory=list,
        description="Ontology schema ids this fact instantiates.",
    )


class OntologySchemaRecord(BaseModel):
    """Schema or type pattern with extraction frequency (M_ont)."""

    id: str
    label: str = Field(min_length=1)
    extraction_count: int = Field(default=0, ge=0)

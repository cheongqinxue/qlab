"""Pydantic schemas for MemGraphRAG LLM structured outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SchemaItem(BaseModel):
    """Ontology schema extracted from a passage."""

    id: str = Field(description="Stable schema id, e.g. s_person_birth_year")
    label: str = Field(description="Human-readable schema/type name")


class FactItem(BaseModel):
    """Concrete fact grounded in the passage."""

    id: str = Field(description="Stable fact id, e.g. f_newton_1642")
    text: str = Field(min_length=1)
    passage_ids: list[str] = Field(default_factory=list)
    schema_ids: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """A_ext joint extraction output."""

    schemas: list[SchemaItem] = Field(default_factory=list)
    facts: list[FactItem] = Field(default_factory=list)


class ConflictGroupItem(BaseModel):
    """One hard conflict cluster."""

    target_id: str = Field(description="Newly added fact id under scrutiny")
    related_ids: list[str] = Field(default_factory=list)
    kind: str = Field(default="hard")
    description: str = Field(default="")


class ConflictDetectionResult(BaseModel):
    """A_det output."""

    groups: list[ConflictGroupItem] = Field(default_factory=list)


class ResolutionResult(BaseModel):
    """A_res adjudication plan."""

    drop: list[str] = Field(default_factory=list, description="Fact ids to remove")
    keep: list[str] = Field(default_factory=list, description="Fact ids to retain")
    rationale: str = Field(default="")

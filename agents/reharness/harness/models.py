"""Data models for ReHarness reflection bank entries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MemoryAgentOutput(BaseModel):
    """Structured output for πM (forward reflection write)."""

    reflection: str = Field(
        description=(
            "Concise reflection for similar future tasks: bullet lists for "
            "operational evidence and supported strategies."
        ),
    )


class RevisionAgentOutput(BaseModel):
    """Structured output for πR (revise an existing reflection)."""

    revised_reflection: str = Field(
        description="Updated reflection text after incorporating the latest run.",
    )
    guidance: str = Field(
        default="",
        description="Bullet guidance on when this reflection applies; empty if unchanged.",
    )


class ReflectionPayload(BaseModel):
    """Stored reflection and its source instance."""

    reflection: str = Field(description="Lesson text for future tasks.")
    guidance: str | None = Field(
        default=None,
        description="Optional bullet guidance on when this reflection applies.",
    )
    source_task: str = Field(description="User task text for the source instance.")
    trajectory_text: str = Field(
        description="Serialized agent trajectory from the source instance.",
    )
    user_feedback: str = Field(
        description="Qualitative feedback from the user after the instance.",
    )


def reflection_search_text(payload: ReflectionPayload) -> str:
    """Text used for dense retrieval indexing."""
    parts = [payload.reflection, payload.source_task]
    if payload.guidance:
        parts.append(payload.guidance)
    return "\n".join(parts)


def payload_from_memory_content(content: Any) -> ReflectionPayload:
    """Parse Qdrant payload content into ``ReflectionPayload``."""
    if isinstance(content, str):
        return ReflectionPayload.model_validate_json(content)
    if isinstance(content, dict):
        return ReflectionPayload.model_validate(content)
    msg = f"Unsupported reflection content type: {type(content)!r}"
    raise TypeError(msg)

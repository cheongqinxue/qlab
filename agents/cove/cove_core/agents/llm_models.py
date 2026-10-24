"""Structured LLM outputs for CoVe steps."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VerificationPlan(BaseModel):
    """Planner output: ordered verification questions."""

    questions: list[str] = Field(min_length=1, description="Short factual questions")

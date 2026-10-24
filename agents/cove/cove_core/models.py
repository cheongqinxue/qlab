"""Structured CoVe run payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VerificationPair(BaseModel):
    """One verification question and its isolated answer."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class CoVeRunResult(BaseModel):
    """Full trace of a Chain-of-Verification run."""

    query: str
    query_id: str | None = None
    baseline_response: str
    verification_questions: list[str]
    verification_pairs: list[VerificationPair]
    final_response: str
    execution_mode: str = "factored"

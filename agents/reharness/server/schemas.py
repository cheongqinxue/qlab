"""HTTP request and response schemas for ReHarness."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthOut(BaseModel):
    """GET /health payload."""

    status: str
    experiment: str
    dependencies: dict[str, str]


class CreateTaskRequest(BaseModel):
    """POST /tasks body — start a new task instance."""

    task: str = Field(min_length=1, max_length=32_000)


class CreateTaskResponse(BaseModel):
    """POST /tasks response after the agent reaches feedback or completion."""

    task_id: str
    status: str
    assistant_message: str | None = None
    interrupt: dict[str, Any] | None = None


class FeedbackRequest(BaseModel):
    """POST /tasks/{task_id}/feedback body — qualitative user feedback."""

    feedback: str = Field(min_length=1, max_length=16_000)


class TaskStatusResponse(BaseModel):
    """Task checkpoint snapshot."""

    task_id: str
    status: str
    assistant_message: str | None = None
    harness_complete: bool = False


class ResetMemoryResponse(BaseModel):
    """POST /memory/reset response."""

    cleared: bool = Field(description="True when all points were removed from the bank.")

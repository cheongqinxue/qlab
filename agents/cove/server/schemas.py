"""HTTP response schemas for CoVe."""

from __future__ import annotations

from pydantic import BaseModel

from cove_core.models import CoVeRunResult


class HealthOut(BaseModel):
    """GET /health payload."""

    status: str
    experiment: str
    agents_mode: str
    execution_mode: str


class CoVeBootstrapOut(CoVeRunResult):
    """POST /eval/bootstrap-toy payload."""

    source: str

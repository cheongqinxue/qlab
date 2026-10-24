"""Health checks."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState
from server.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    """Service liveness and agent mode."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    return HealthOut(
        status="ok",
        experiment="cove",
        agents_mode=state.runtime.agents_mode,
        execution_mode=state.runtime.config.verification_execution_mode,
    )

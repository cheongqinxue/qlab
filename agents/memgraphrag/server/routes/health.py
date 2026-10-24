"""Health checks."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState
from server.schemas import HealthOut, MemoryCountsOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    """Service liveness and global memory tier counts."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    mem = state.memory
    return HealthOut(
        status="ok",
        experiment="memgraphrag",
        memory=MemoryCountsOut(
            ontology=mem.ontology_count,
            facts=mem.fact_count,
            passages=mem.passage_count,
        ),
    )

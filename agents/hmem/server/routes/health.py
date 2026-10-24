"""Health checks."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState
from server.schemas import HealthOut, HybridCountsOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    """Service liveness and hybrid index stats."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    store = state.runtime.store
    return HealthOut(
        status="ok",
        experiment="hmem",
        agents_mode=state.runtime.agents_mode,
        hybrid=HybridCountsOut(
            fragments=store.fragment_count,
            leaves=store.leaf_count,
            entities=store.entity_count,
            summary_nodes=store.summary_nodes,
            indexed=store.indexed,
        ),
    )

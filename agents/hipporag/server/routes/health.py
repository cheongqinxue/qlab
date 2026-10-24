"""Health checks."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState
from server.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    """Service liveness and graph index stats."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    store = state.runtime.store
    return HealthOut(
        status="ok",
        experiment="hipporag",
        passages=store.passage_count,
        entities=store.entity_count,
        triples=store.triple_count,
        rel_edges=store.rel_edge_count,
        synonym_edges=store.synonym_edge_count,
    )

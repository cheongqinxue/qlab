"""Health checks."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    """Service liveness and graph index stats."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    store = state.runtime.store
    return {
        "status": "ok",
        "experiment": "hipporag",
        "passages": store.passage_count,
        "entities": store.entity_count,
        "triples": store.triple_count,
        "rel_edges": store.rel_edge_count,
        "synonym_edges": store.synonym_edge_count,
    }

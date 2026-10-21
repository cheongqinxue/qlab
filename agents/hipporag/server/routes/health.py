"""Health checks."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    """Service liveness and graph index stats."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    return {
        "status": "ok",
        "experiment": "hipporag",
        "passages": state.graph.passage_count,
        "triples": state.graph.triple_count,
    }

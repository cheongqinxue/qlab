"""Health checks."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    """Service liveness and global memory tier counts."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    mem = state.memory
    return {
        "status": "ok",
        "experiment": "memgraphrag",
        "memory": {
            "ontology": mem.ontology_count,
            "facts": mem.fact_count,
            "passages": mem.passage_count,
        },
    }

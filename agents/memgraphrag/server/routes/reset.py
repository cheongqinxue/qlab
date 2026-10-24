"""Reset global memory and derived graph snapshots."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState

router = APIRouter(tags=["admin"])


@router.post("/reset")
async def reset_memory(request: Request) -> dict[str, str]:
    """Clear all memory tiers and optional Neo4j projection."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    await state.memory.aclear()
    if state.runtime.neo4j is not None:
        state.runtime.neo4j.clear()
    return {"status": "cleared"}

"""Reset global memory and derived graph snapshots."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState
from server.schemas import ResetOut

router = APIRouter(tags=["admin"])


@router.post("/reset", response_model=ResetOut)
async def reset_memory(request: Request) -> ResetOut:
    """Clear all memory tiers and optional Neo4j projection."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    await state.memory.aclear()
    if state.runtime.neo4j is not None:
        state.runtime.neo4j.clear()
    return ResetOut(status="cleared")

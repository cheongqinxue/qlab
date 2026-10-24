"""Reset hybrid memory."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState
from server.schemas import ResetOut

router = APIRouter(tags=["reset"])


@router.post("/reset", response_model=ResetOut)
async def reset_memory(request: Request) -> ResetOut:
    """Clear all fragments, events, and graph edges."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    state.runtime.store.clear()
    state.runtime.fixture_query = None
    state.runtime.fixture_answer = None
    return ResetOut(status="cleared")

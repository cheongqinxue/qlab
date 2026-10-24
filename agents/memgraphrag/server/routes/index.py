"""Indexing route — run multi-agent graph construction."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState

router = APIRouter(tags=["index"])


@router.post("/index")
async def run_index(request: Request) -> dict[str, object]:
    """Build/update hierarchical graph from staged corpus."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    return await state.runtime.orchestrator.run_full_index()

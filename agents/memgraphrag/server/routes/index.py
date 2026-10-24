"""Indexing route — run multi-agent graph construction."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app_state import AppState
from server.schemas import IndexOut

router = APIRouter(tags=["index"])


@router.post("/index", response_model=IndexOut)
async def run_index(request: Request) -> IndexOut:
    """Build/update hierarchical graph from staged corpus."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    summary = await state.runtime.orchestrator.run_full_index()
    return IndexOut(**summary)

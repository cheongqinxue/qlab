"""Hybrid index maintenance."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from hmem_core.indexing.fixture_indexer import run_consolidation_pass
from server.app_state import AppState
from server.schemas import IndexOut

router = APIRouter(tags=["index"])


@router.post("/index", response_model=IndexOut)
async def run_index(request: Request) -> IndexOut:
    """Run consolidation pass (scaffold) over current leaf events."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    if not state.runtime.store.events:
        raise HTTPException(status_code=400, detail="No fragments to index")
    runtime = state.runtime
    summaries = run_consolidation_pass(
        runtime.store,
        config=runtime.config,
        encoder=runtime.encoder,
        agents_mode=runtime.agents_mode,
    )
    runtime.refresh_engine()
    return IndexOut(summary_nodes=summaries, indexed=state.runtime.store.indexed)

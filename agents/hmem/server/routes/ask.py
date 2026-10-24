"""Memory QA."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from server.app_state import AppState
from server.schemas import AskIn, AskOut

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskOut)
async def ask(body: AskIn, request: Request) -> AskOut:
    """Answer a query via the retrieval engine (fixture or LLM)."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    runtime = state.runtime
    if not runtime.store.fragment_count:
        raise HTTPException(status_code=400, detail="Memory is empty; bootstrap or ingest first")
    assert runtime.engine is not None
    try:
        result = runtime.engine.ask(body.query)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return AskOut.model_validate(result.model_dump())

"""Retrieval routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from server.app_state import AppState

router = APIRouter(tags=["retrieve"])


class RetrieveIn(BaseModel):
    """Retrieval request."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


@router.post("/retrieve")
async def retrieve(body: RetrieveIn, request: Request) -> dict[str, object]:
    """Memory-guided passage retrieval."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    hits = await state.runtime.retriever.retrieve(body.query, top_k=body.top_k)
    return {
        "query": body.query,
        "hits": [h.model_dump() for h in hits],
    }

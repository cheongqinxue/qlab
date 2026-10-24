"""Retrieval routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from server.app_state import AppState
from server.schemas import RetrieveOut

router = APIRouter(tags=["retrieve"])


class RetrieveIn(BaseModel):
    """Retrieval request."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


@router.post("/retrieve", response_model=RetrieveOut)
async def retrieve(body: RetrieveIn, request: Request) -> RetrieveOut:
    """Memory-guided passage retrieval."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    hits = await state.runtime.retriever.retrieve(body.query, top_k=body.top_k)
    return RetrieveOut(query=body.query, hits=hits)

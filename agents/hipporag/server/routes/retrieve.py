"""Retrieval routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from server.app_state import AppState
from server.openie_factory import get_openie_extractor
from server.schemas import RetrieveOut

router = APIRouter(tags=["retrieve"])


class RetrieveIn(BaseModel):
    """Retrieval request."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


@router.post("/retrieve", response_model=RetrieveOut)
async def retrieve(body: RetrieveIn, request: Request) -> RetrieveOut:
    """Rank passages with HippoRAG PPR."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    extractor = get_openie_extractor()
    entities = extractor.extract_query_entities(body.query)
    hits, debug = state.runtime.retriever.retrieve_with_debug(
        query=body.query,
        query_entities=entities,
        top_k=body.top_k,
    )
    return RetrieveOut(
        query=body.query,
        query_entities=entities,
        seed_entities=list(debug.seed_entities),
        hits=hits,
    )

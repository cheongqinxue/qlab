"""Retrieval routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from server.app_state import AppState
from server.openie_factory import get_openie_extractor

router = APIRouter(tags=["retrieve"])


class RetrieveIn(BaseModel):
    """Retrieval request."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


@router.post("/retrieve")
async def retrieve(body: RetrieveIn, request: Request) -> dict[str, object]:
    """Rank passages with HippoRAG PPR."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    extractor = get_openie_extractor()
    entities = extractor.extract_query_entities(body.query)
    hits, debug = state.runtime.retriever.retrieve_with_debug(
        query=body.query,
        query_entities=entities,
        top_k=body.top_k,
    )
    return {
        "query": body.query,
        "query_entities": entities,
        "seed_entities": debug.seed_entities,
        "hits": [h.model_dump() for h in hits],
    }

"""Corpus ingest routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from hipporag_core.models import Passage
from hipporag_core.toy_bootstrap import index_toy_graph, load_toy_passages
from server.app_state import AppState

router = APIRouter(prefix="/corpus", tags=["corpus"])


class PassageIn(BaseModel):
    """Request body for one passage."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


@router.post("/passages")
async def add_passage(body: PassageIn, request: Request) -> dict[str, object]:
    """Append or update a passage (indexed on ``POST /index``)."""
    app_state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    app_state.runtime.store.stage_passage(Passage(id=body.id, text=body.text))
    return {"passage_id": body.id, "passages": app_state.runtime.store.passage_count}


@router.post("/load-toy")
async def load_toy_corpus_route(request: Request) -> dict[str, object]:
    """Stage the 5-passage ``eval/toy_multihop`` corpus (no LLM)."""
    app_state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    count = load_toy_passages(app_state.runtime)
    return {"source": "eval/toy_multihop/corpus.json", "passages": count}


@router.post("/bootstrap-toy")
async def bootstrap_toy_route(request: Request) -> dict[str, object]:
    """Load toy corpus and index with fixture OpenIE → Neo4j + memory graph."""
    app_state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    load_toy_passages(app_state.runtime)
    summary = index_toy_graph(app_state.runtime)
    return {"source": "eval/toy_multihop", "openie": "fixture", **summary}

"""Indexing routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from hipporag_core.models import OpenIEExtraction
from server.app_state import AppState
from server.openie_factory import get_openie_extractor
from server.schemas import IndexOut, StatusOut

router = APIRouter(tags=["index"])


@router.post("/index", response_model=IndexOut)
async def run_index(request: Request) -> IndexOut:
    """Run OpenIE and graph indexing for pending passages."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    extractor = get_openie_extractor()
    pending = state.runtime.store.pending_passage_ids()
    openie_map: dict[str, OpenIEExtraction] = {}
    for passage_id in pending:
        text = state.runtime.store.get_passage_text(passage_id)
        openie_map[passage_id] = extractor.extract_passage(passage_id, text)
    indexed = state.runtime.indexer.index_passages(openie_by_passage=openie_map)
    synonyms = state.runtime.indexer.add_synonymy_edges()
    store = state.runtime.store
    return IndexOut(
        indexed_passages=indexed,
        synonym_edges_added=synonyms,
        passages=store.passage_count,
        entities=store.entity_count,
        triples=store.triple_count,
    )


@router.post("/reset", response_model=StatusOut)
async def reset_index(request: Request) -> StatusOut:
    """Clear graph index (passage bodies remain staged)."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    state.runtime.store.clear()
    return StatusOut(status="cleared")

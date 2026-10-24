"""Corpus ingest and toy bootstrap."""

from __future__ import annotations

from fastapi import APIRouter, Request

from hmem_core.indexing.fixture_indexer import build_indexer
from hmem_core.models import MemoryFragment
from hmem_core.toy_bootstrap import bootstrap_toy
from server.app_state import AppState
from server.schemas import BootstrapOut, FragmentBatchOut, FragmentIn

router = APIRouter(prefix="/corpus", tags=["corpus"])


def _state(request: Request) -> AppState:
    return request.app.state.runtime  # type: ignore[attr-defined]


@router.post("/fragments", response_model=FragmentBatchOut)
async def add_fragment(body: FragmentIn, request: Request) -> FragmentBatchOut:
    """Ingest a single memory fragment as a leaf event."""
    app_state = _state(request)
    runtime = app_state.runtime
    fragment = MemoryFragment.model_validate(body.model_dump())
    indexer = build_indexer(
        runtime.store,
        runtime.encoder,
        agents_mode=runtime.agents_mode,
    )
    added = indexer.index_fragment(fragment)
    return FragmentBatchOut(fragments=runtime.store.fragment_count, leaves_added=added)


@router.post("/bootstrap-toy", response_model=BootstrapOut)
async def bootstrap_toy_route(request: Request) -> BootstrapOut:
    """Load ``eval/toy_locomo_mini`` and run scaffold indexing."""
    app_state = _state(request)
    summary = await bootstrap_toy(app_state.runtime)
    return BootstrapOut.model_validate(summary)

"""Corpus staging routes — populate M_pas before indexing."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from memgraphrag_core.toy_bootstrap import index_toy_memory, load_toy_passages
from server.app_state import AppState

router = APIRouter(prefix="/corpus", tags=["corpus"])


class PassageIn(BaseModel):
    """One passage body."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


@router.post("/passages")
async def add_passage(body: PassageIn, request: Request) -> dict[str, object]:
    """Stage a passage in M_pas (indexed on ``POST /index``)."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    passage_id = await state.memory.add({"id": body.id, "text": body.text})
    return {"passage_id": passage_id, "passages": state.memory.passage_count}


@router.post("/load-toy")
async def load_toy(request: Request) -> dict[str, object]:
    """Stage ``eval/toy_conflict`` passages without indexing."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    count = await load_toy_passages(state.runtime)
    return {"source": "eval/toy_conflict/corpus.json", "passages": count}


@router.post("/bootstrap-toy")
async def bootstrap_toy(request: Request) -> dict[str, object]:
    """Stage toy corpus and run fixture index pipeline."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    await load_toy_passages(state.runtime)
    summary = await index_toy_memory(state.runtime)
    return {"source": "eval/toy_conflict", "extraction": "fixture", **summary}

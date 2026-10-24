"""Toy eval bootstrap routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from cove_core.eval_loader import load_toy_manifest
from server.app_state import AppState

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/bootstrap-toy")
async def bootstrap_toy(request: Request) -> dict[str, object]:
    """Run the committed Mexican–American War CoVe fixture."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    manifest = load_toy_manifest()
    query_id = str(manifest["query_id"])
    query = str(manifest["query"])
    result = await state.runtime.orchestrator.run(query, query_id=query_id)
    return {"source": "eval/toy_mexican_american_war", **result.model_dump()}

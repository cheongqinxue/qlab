"""Reflection bank maintenance."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from server.app_state import AppState

router = APIRouter(prefix="/memory", tags=["memory"])


class ResetMemoryResponse(BaseModel):
    """Result of clearing the Qdrant reflection collection."""

    cleared: bool = Field(description="True when all points were removed from the bank.")


def _memory(request: Request):
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    return state.memory


@router.post("/reset", response_model=ResetMemoryResponse)
async def reset_memory_bank(request: Request) -> ResetMemoryResponse:
    """Delete every reflection in the Qdrant collection (fresh experiment)."""
    memory = _memory(request)
    await memory.aclear()
    return ResetMemoryResponse(cleared=True)

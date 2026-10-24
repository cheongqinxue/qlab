"""Reflection bank maintenance."""

from __future__ import annotations

from fastapi import APIRouter, Request

from agents_utils.memory.base_memory import BaseMemory

from server.app_state import AppState
from server.schemas import ResetMemoryResponse

router = APIRouter(prefix="/memory", tags=["memory"])


def _memory(request: Request) -> BaseMemory:
    """Return the Qdrant reflection bank from app state."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    return state.memory


@router.post("/reset", response_model=ResetMemoryResponse)
async def reset_memory_bank(request: Request) -> ResetMemoryResponse:
    """Delete every reflection in the Qdrant collection (fresh experiment)."""
    memory = _memory(request)
    await memory.aclear()
    return ResetMemoryResponse(cleared=True)

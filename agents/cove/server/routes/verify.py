"""Chain-of-Verification run endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from cove_core.eval_loader import load_toy_manifest
from server.app_state import AppState

router = APIRouter(tags=["verify"])


class VerifyIn(BaseModel):
    """CoVe run request."""

    query: str | None = Field(default=None, min_length=1)
    query_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_query_or_id(self) -> VerifyIn:
        """Ensure at least one of query or query_id is set."""
        if not self.query and not self.query_id:
            msg = "Provide query or query_id"
            raise ValueError(msg)
        return self


@router.post("/verify")
async def verify(body: VerifyIn, request: Request) -> dict[str, object]:
    """Run baseline → plan → isolated verify → revise."""
    state: AppState = request.app.state.runtime  # type: ignore[attr-defined]
    query = body.query
    query_id = body.query_id
    if query_id == "mexican_american_war" and not query:
        query = str(load_toy_manifest()["query"])
    if query is None:
        raise HTTPException(status_code=400, detail="query is required when query_id is unknown")
    if state.runtime.agents_mode == "fixture" and query_id is None:
        raise HTTPException(
            status_code=400,
            detail="Fixture mode requires query_id (e.g. mexican_american_war)",
        )
    result = await state.runtime.orchestrator.run(query, query_id=query_id)
    return result.model_dump()

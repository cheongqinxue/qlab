"""Health checks."""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter

from server.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    """Agent liveness and Qdrant reachability."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333").rstrip("/")
    deps: dict[str, str] = {"agent": "ok"}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            qdrant = await client.get(f"{qdrant_url}/readyz")
            deps["qdrant"] = "ok" if qdrant.status_code == 200 else "error"
    except httpx.HTTPError:
        deps["qdrant"] = "unreachable"
    return HealthOut(status="ok", experiment="reharness", dependencies=deps)

"""Health checks."""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, object]:
    """Agent liveness and Qdrant reachability."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333").rstrip("/")
    deps: dict[str, str] = {"agent": "ok"}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            qdrant = await client.get(f"{qdrant_url}/readyz")
            deps["qdrant"] = "ok" if qdrant.status_code == 200 else "error"
    except httpx.HTTPError:
        deps["qdrant"] = "unreachable"
    return {"status": "ok", "dependencies": deps}

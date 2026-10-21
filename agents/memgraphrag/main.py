"""FastAPI entrypoint for the MemGraphRAG experiment (scaffold)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agents_utils.env import load_agents_env
from agents_utils.logging_setup import configure_agent_logging
from agents_utils.observability.langfuse_tracing import (
    ensure_langfuse_client,
    flush_langfuse,
    langfuse_tracing_enabled,
)
from memgraphrag_core.global_memory import GlobalMemory
from server.app_state import AppState
from server.routes import health
from server.settings import get_settings

load_agents_env()
_log_file = configure_agent_logging()
logger = logging.getLogger(__name__)
if _log_file is not None:
    logger.info("File logging enabled at %s", _log_file)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize global memory and optional Langfuse."""
    settings = get_settings()
    app.state.runtime = AppState(memory=GlobalMemory())
    if langfuse_tracing_enabled():
        ensure_langfuse_client()
        logger.info("Langfuse tracing enabled")
    logger.info("MemGraphRAG scaffold ready (port=%s)", settings.agent_port)
    yield
    flush_langfuse()


app = FastAPI(title="MemGraphRAG Agent", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)


def run() -> None:
    """Run uvicorn (local dev)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.agent_port,
        reload=False,
    )


if __name__ == "__main__":
    run()

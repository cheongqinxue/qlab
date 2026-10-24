"""FastAPI entrypoint for the MemGraphRAG experiment."""

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
from memgraphrag_core.memory.config import MemGraphMemoryConfig
from memgraphrag_core.runtime import build_runtime
from memgraphrag_core.toy_bootstrap import bootstrap_toy_conflict
from server.app_state import AppState
from server.routes import corpus, health, index, reset, retrieve
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
    mem_config = MemGraphMemoryConfig.from_file(settings.config_path)
    runtime = build_runtime(mem_config)
    app.state.runtime = AppState(runtime=runtime)
    bootstrap = await bootstrap_toy_conflict(runtime)
    if bootstrap is not None:
        logger.info(
            "Toy conflict corpus indexed (passages=%s facts=%s conflicts_resolved=%s)",
            bootstrap.get("passages"),
            bootstrap.get("facts"),
            bootstrap.get("conflicts_resolved"),
        )
    if langfuse_tracing_enabled():
        ensure_langfuse_client()
        logger.info("Langfuse tracing enabled")
    logger.info("MemGraphRAG ready (port=%s)", settings.agent_port)
    yield
    runtime.close()
    flush_langfuse()


app = FastAPI(title="MemGraphRAG Agent", version="0.2.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(corpus.router)
app.include_router(index.router)
app.include_router(retrieve.router)
app.include_router(reset.router)


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

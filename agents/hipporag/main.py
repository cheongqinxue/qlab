"""FastAPI entrypoint for the HippoRAG experiment (scaffold)."""

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
from hipporag_core.runtime import build_runtime
from hipporag_core.toy_bootstrap import bootstrap_toy_enabled, bootstrap_toy_multihop
from server.app_state import AppState
from server.routes import ask, corpus, health, index, retrieve
from server.settings import get_settings

load_agents_env()
_log_file = configure_agent_logging()
logger = logging.getLogger(__name__)
if _log_file is not None:
    logger.info("File logging enabled at %s", _log_file)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize graph store and optional Langfuse."""
    settings = get_settings()
    runtime = build_runtime(
        damping=settings.ppr_damping,
        synonymy_threshold=settings.synonymy_threshold,
    )
    app.state.runtime = AppState(runtime=runtime)
    if bootstrap_toy_enabled():
        bootstrap = bootstrap_toy_multihop(runtime)
        if bootstrap is not None:
            logger.info(
                "Toy corpus bootstrapped into graph (passages=%s entities=%s triples=%s)",
                bootstrap.get("passages"),
                bootstrap.get("entities"),
                bootstrap.get("triples"),
            )
    else:
        logger.info(
            "Toy bootstrap skipped (HIPPORAG_BOOTSTRAP_TOY=0); "
            "use POST /corpus/bootstrap-toy or Streamlit sidebar"
        )
    if langfuse_tracing_enabled():
        ensure_langfuse_client()
        logger.info("Langfuse tracing enabled")
    logger.info("HippoRAG ready (port=%s)", settings.agent_port)
    yield
    runtime.close()
    flush_langfuse()


app = FastAPI(title="HippoRAG Agent", version="0.2.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(corpus.router)
app.include_router(index.router)
app.include_router(retrieve.router)
app.include_router(ask.router)


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

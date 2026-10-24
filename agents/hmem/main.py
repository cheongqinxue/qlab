"""FastAPI entrypoint for the H-MEM experiment."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agents_utils.env import load_agents_env
from agents_utils.logging_setup import configure_agent_logging
from agents_utils.observability.langfuse_tracing import (
    ensure_langfuse_client,
    flush_langfuse,
    langfuse_tracing_enabled,
)
from hmem_core.config import HMemConfig
from hmem_core.runtime import build_runtime
from hmem_core.toy_bootstrap import bootstrap_toy
from server.app_state import AppState
from server.routes import ask, corpus, health, index, reset
from server.settings import get_settings

load_agents_env()
_log_file = configure_agent_logging()
logger = logging.getLogger(__name__)
if _log_file is not None:
    logger.info("File logging enabled at %s", _log_file)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize hybrid memory runtime and optional Langfuse."""
    settings = get_settings()
    config = HMemConfig.from_file(settings.config_path)
    hmem_root = settings.config_path.resolve().parent.parent
    memory_dir = config.memory_data_dir
    if not memory_dir.startswith("/"):
        config = config.model_copy(update={"memory_data_dir": str(hmem_root / memory_dir)})
    runtime = build_runtime(config, agents_mode=settings.agents_mode)
    app.state.runtime = AppState(runtime=runtime)
    if settings.bootstrap_toy:
        summary = await bootstrap_toy(runtime)
        logger.info("Toy bootstrap indexed: %s", summary)
    if langfuse_tracing_enabled():
        ensure_langfuse_client()
        logger.info("Langfuse tracing enabled")
    os.environ.setdefault("AGENT_NAME", "hmem")
    logger.info("H-MEM ready (port=%s mode=%s)", settings.agent_port, settings.agents_mode)
    yield
    flush_langfuse()


app = FastAPI(title="H-MEM Agent", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(corpus.router)
app.include_router(index.router)
app.include_router(ask.router)
app.include_router(reset.router)


def run() -> None:
    """Run uvicorn for local development."""
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

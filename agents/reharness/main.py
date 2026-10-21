"""FastAPI entrypoint for the ReHarness task agent."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents_utils.env import load_agents_env
from agents_utils.logging_setup import configure_agent_logging
from agents_utils.memory.langchain_qdrant_memory import QdrantMemory, QdrantMemoryConfig
from agents_utils.observability.langfuse_tracing import (
    ensure_langfuse_client,
    flush_langfuse,
    langfuse_tracing_enabled,
)
from agents_utils.memory.sentence_transformers_encoder import SentenceTransformersEncoder
from graph.build import build_task_graph
from harness.reharness import ReHarness
from server.app_state import AppState
from server.routes import health, memory, tasks
from server.settings import get_settings
from tools.python_runner import run_python

load_agents_env()
_log_file = configure_agent_logging()
logger = logging.getLogger(__name__)
if _log_file is not None:
    logger.info("File logging enabled at %s", _log_file)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize memory, harness, and LangGraph checkpointer."""
    settings = get_settings()
    os.environ.setdefault(
        "PYTHON_RUN_TIMEOUT_SECONDS",
        str(settings.python_run_timeout_seconds),
    )

    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Loading embedding model %s", settings.embedding_model)
    encoder = SentenceTransformersEncoder(settings.embedding_model)

    config = QdrantMemoryConfig.from_file(settings.qdrant_memory_config)
    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url:
        config = config.model_copy(update={"url": qdrant_url})

    memory = QdrantMemory.build_or_restore(config, encoder=encoder)
    llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
    reharness = ReHarness(memory, llm=llm, top_k=settings.reharness_top_k)

    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_path)) as checkpointer:
        graph = build_task_graph(
            llm=llm,
            reharness=reharness,
            tools=[run_python],
            checkpointer=checkpointer,
        )
        app.state.runtime = AppState(graph=graph, checkpointer=checkpointer, memory=memory)
        if langfuse_tracing_enabled():
            ensure_langfuse_client()
            logger.info("Langfuse tracing enabled")
        logger.info("ReHarness agent ready (model=%s)", settings.openai_model)
        yield
        flush_langfuse()


app = FastAPI(title="ReHarness Agent", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(memory.router)
app.include_router(tasks.router)


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

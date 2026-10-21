"""Langfuse tracing helpers for LangGraph / LangChain agents."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

_client_ready = False


def langfuse_host() -> str:
    """Langfuse server URL for the Python SDK."""
    return (
        os.environ.get("LANGFUSE_BASE_URL")
        or os.environ.get("LANGFUSE_HOST")
        or "https://cloud.langfuse.com"
    ).rstrip("/")


def ensure_langfuse_client() -> None:
    """Initialize the process-wide Langfuse client from environment (idempotent)."""
    global _client_ready
    if _client_ready or not langfuse_tracing_enabled():
        return
    from langfuse import Langfuse

    Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=langfuse_host(),
    )
    _client_ready = True
    logger.info("Langfuse client initialized (host=%s)", langfuse_host())


def langfuse_trace_tags() -> list[str]:
    """Tags sent with each trace (experiment / agent differentiation)."""
    tags: list[str] = []
    agent = os.environ.get("AGENT_NAME", "").strip()
    if agent:
        tags.append(agent)
    extra = os.environ.get("LANGFUSE_TAGS", "").strip()
    if extra:
        tags.extend(part.strip() for part in extra.split(",") if part.strip())
    return tags or ["agent"]


def langfuse_tracing_enabled() -> bool:
    """Return True when Langfuse env is configured and tracing is not disabled."""
    flag = os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower()
    if flag in ("0", "false", "no", "off"):
        return False
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def create_langfuse_callback_handler() -> CallbackHandler | None:
    """Build a LangChain callback handler when Langfuse credentials are present."""
    if not langfuse_tracing_enabled():
        return None
    ensure_langfuse_client()
    return CallbackHandler(public_key=os.environ["LANGFUSE_PUBLIC_KEY"])


def build_graph_invoke_config(task_id: str) -> dict[str, Any]:
    """LangGraph invoke config with thread id and optional Langfuse callbacks."""
    config: dict[str, Any] = {"configurable": {"thread_id": task_id}}
    handler = create_langfuse_callback_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    return config


@contextmanager
def langfuse_task_session(*, task_id: str, task_preview: str) -> Iterator[None]:
    """Attach session metadata to traces for one task instance."""
    if not langfuse_tracing_enabled():
        yield
        return
    ensure_langfuse_client()
    from langfuse import propagate_attributes

    preview = task_preview[:200]
    metadata: dict[str, str] = {"task_preview": preview}
    if agent := os.environ.get("AGENT_NAME", "").strip():
        metadata["agent_name"] = agent
    with propagate_attributes(
        session_id=task_id,
        tags=langfuse_trace_tags(),
        metadata=metadata,
    ):
        yield


def flush_langfuse() -> None:
    """Flush pending Langfuse events (call after each task and on shutdown)."""
    if not langfuse_tracing_enabled():
        return
    try:
        ensure_langfuse_client()
        from langfuse import get_client

        get_client(public_key=os.environ["LANGFUSE_PUBLIC_KEY"]).flush()
    except Exception:
        logger.exception("Langfuse flush failed")

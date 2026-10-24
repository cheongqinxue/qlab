"""Environment-backed settings for the agent server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Configuration for the ReHarness agent server.

    Implements :class:`agents_utils.web_tools_settings.WebToolsSettings` for
    optional ``websearch`` / ``webfetch`` tools when ``web_tools_enabled``.
    """

    openai_api_key: str
    openai_model: str
    qdrant_memory_config: Path
    embedding_model: str
    reharness_top_k: int
    agent_port: int
    python_run_timeout_seconds: int
    checkpoint_path: Path
    web_tools_enabled: bool
    websearch_provider: str
    langsearch_api_key: str
    webfetch_timeout_seconds: float
    webfetch_network_idle_timeout_ms: int
    webfetch_post_load_wait_ms: int
    webfetch_max_chars: int
    api_retry_max_attempts: int


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment variables.

    Returns:
        Cached :class:`Settings` instance.

    Raises:
        RuntimeError: When ``OPENAI_API_KEY`` is missing.
    """
    from agents_utils.env import load_agents_env

    load_agents_env()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        msg = "OPENAI_API_KEY is required"
        raise RuntimeError(msg)
    web_tools_flag = os.environ.get("REHARNESS_WEB_TOOLS", "0").lower()
    return Settings(
        openai_api_key=api_key,
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        qdrant_memory_config=Path(
            os.environ.get(
                "QDRANT_MEMORY_CONFIG",
                "/app/agents/reharness/config/qdrant_memory.json",
            )
        ),
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        reharness_top_k=int(os.environ.get("REHARNESS_TOP_K", "5")),
        agent_port=int(os.environ.get("AGENT_PORT", "8080")),
        python_run_timeout_seconds=int(os.environ.get("PYTHON_RUN_TIMEOUT_SECONDS", "30")),
        checkpoint_path=Path(os.environ.get("CHECKPOINT_PATH", "/data/checkpoints.sqlite")),
        web_tools_enabled=web_tools_flag in ("1", "true", "yes", "on"),
        websearch_provider=os.environ.get("WEBSEARCH_PROVIDER", "langsearch"),
        langsearch_api_key=os.environ.get("LANGSEARCH_API_KEY", "").strip(),
        webfetch_timeout_seconds=float(os.environ.get("WEBFETCH_TIMEOUT_SECONDS", "30")),
        webfetch_network_idle_timeout_ms=int(
            os.environ.get("WEBFETCH_NETWORK_IDLE_TIMEOUT_MS", "5000")
        ),
        webfetch_post_load_wait_ms=int(os.environ.get("WEBFETCH_POST_LOAD_WAIT_MS", "500")),
        webfetch_max_chars=int(os.environ.get("WEBFETCH_MAX_CHARS", "12000")),
        api_retry_max_attempts=int(os.environ.get("API_RETRY_MAX_ATTEMPTS", "3")),
    )

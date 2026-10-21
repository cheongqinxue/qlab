"""Environment-backed settings for the agent server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Configuration for the ReHarness agent server."""

    openai_api_key: str
    openai_model: str
    qdrant_memory_config: Path
    embedding_model: str
    reharness_top_k: int
    agent_port: int
    python_run_timeout_seconds: int
    checkpoint_path: Path


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment variables."""
    from agents_utils.env import load_agents_env

    load_agents_env()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        msg = "OPENAI_API_KEY is required"
        raise RuntimeError(msg)
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
    )

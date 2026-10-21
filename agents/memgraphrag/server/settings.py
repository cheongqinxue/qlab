"""Environment-backed settings for the MemGraphRAG service."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Configuration for the MemGraphRAG service."""

    agent_port: int
    config_path: Path
    retrieval_top_k: int
    ppr_damping: float


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment and optional JSON config."""
    from agents_utils.env import load_agents_env

    load_agents_env()
    config_path = Path(
        os.environ.get(
            "MEMGRAPHRAG_CONFIG",
            "/app/agents/memgraphrag/config/default.json",
        )
    )
    defaults: dict[str, object] = {}
    if config_path.is_file():
        defaults = json.loads(config_path.read_text(encoding="utf-8"))
    return Settings(
        agent_port=int(os.environ.get("MEMGRAPHRAG_AGENT_PORT", "8082")),
        config_path=config_path,
        retrieval_top_k=int(
            os.environ.get("MEMGRAPHRAG_TOP_K", defaults.get("retrieval_top_k", 5))
        ),
        ppr_damping=float(
            os.environ.get("MEMGRAPHRAG_PPR_DAMPING", defaults.get("ppr_damping", 0.5))
        ),
    )

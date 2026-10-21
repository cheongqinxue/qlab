"""Environment-backed settings for the HippoRAG agent server."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Configuration for the HippoRAG service."""

    agent_port: int
    config_path: Path
    ppr_damping: float
    retrieval_top_k: int


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment and optional JSON config."""
    from agents_utils.env import load_agents_env

    load_agents_env()
    config_path = Path(
        os.environ.get(
            "HIPPORAG_CONFIG",
            "/app/agents/hipporag/config/default.json",
        )
    )
    defaults: dict[str, object] = {}
    if config_path.is_file():
        defaults = json.loads(config_path.read_text(encoding="utf-8"))
    return Settings(
        agent_port=int(os.environ.get("HIPPORAG_AGENT_PORT", "8081")),
        config_path=config_path,
        ppr_damping=float(os.environ.get("HIPPORAG_PPR_DAMPING", defaults.get("ppr_damping", 0.5))),
        retrieval_top_k=int(
            os.environ.get("HIPPORAG_TOP_K", defaults.get("retrieval_top_k", 5))
        ),
    )

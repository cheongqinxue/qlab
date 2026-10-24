"""Environment-backed settings for the CoVe service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Configuration for the CoVe service."""

    agent_port: int
    config_path: Path


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment."""
    from agents_utils.env import load_agents_env

    load_agents_env()
    config_path = Path(
        os.environ.get(
            "COVE_CONFIG",
            "/app/agents/cove/config/default.json",
        )
    )
    return Settings(
        agent_port=int(os.environ.get("COVE_AGENT_PORT", "8083")),
        config_path=config_path,
    )

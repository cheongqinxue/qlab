"""Environment-backed settings for the H-MEM service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Configuration for the H-MEM agent server."""

    agent_port: int
    config_path: Path
    agents_mode: str
    bootstrap_toy: bool


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment variables.

    Returns:
        Cached settings instance.
    """
    from agents_utils.env import load_agents_env

    load_agents_env()
    hmem_root = Path(__file__).resolve().parents[1]
    default_config = hmem_root / "config" / "default.json"
    config_path = Path(
        os.environ.get("HMEM_CONFIG", str(default_config)),
    )
    if not config_path.is_file():
        config_path = default_config
    bootstrap_flag = os.environ.get("HMEM_BOOTSTRAP_TOY", "0").lower()
    return Settings(
        agent_port=int(os.environ.get("HMEM_AGENT_PORT", "8084")),
        config_path=config_path,
        agents_mode=os.environ.get("HMEM_AGENTS", "fixture").lower(),
        bootstrap_toy=bootstrap_flag in ("1", "true", "yes", "on"),
    )

"""Mutable application state attached to FastAPI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents_utils.memory.base_memory import BaseMemory


@dataclass
class AppState:
    """Runtime handles for the agent server."""

    graph: Any
    checkpointer: Any
    memory: BaseMemory

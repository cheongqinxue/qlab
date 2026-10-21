"""Mutable application state attached to FastAPI."""

from __future__ import annotations

from dataclasses import dataclass

from memgraphrag_core.global_memory import GlobalMemory


@dataclass
class AppState:
    """Runtime handles for the MemGraphRAG server."""

    memory: GlobalMemory

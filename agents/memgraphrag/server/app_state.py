"""Mutable application state attached to FastAPI."""

from __future__ import annotations

from dataclasses import dataclass

from memgraphrag_core.runtime import MemgraphRuntime


@dataclass
class AppState:
    """Runtime handles for the MemGraphRAG server."""

    runtime: MemgraphRuntime

    @property
    def memory(self):
        """Shortcut to global memory bank."""
        return self.runtime.memory

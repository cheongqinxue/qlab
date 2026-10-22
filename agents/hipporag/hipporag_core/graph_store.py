"""Graph store factory (memory or Neo4j)."""

from __future__ import annotations

import os

from hipporag_core.memory_graph_store import MemoryGraphStore
from hipporag_core.neo4j_graph_store import Neo4jGraphStore

GraphStore = MemoryGraphStore


def create_graph_store() -> MemoryGraphStore:
    """Create graph store from ``HIPPORAG_GRAPH_BACKEND`` env.

    Returns:
        ``memory`` (default in tests) or ``neo4j`` for Docker.
    """
    backend = os.environ.get("HIPPORAG_GRAPH_BACKEND", "neo4j").lower()
    if backend == "memory":
        return MemoryGraphStore()
    return Neo4jGraphStore.from_env()

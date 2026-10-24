"""Deprecated alias — use ``memgraphrag_core.memory.MemGraphGlobalMemory`` instead.

The experiment's primary store subclasses ``agents_utils.memory.base_memory.BaseMemory``.
``ThreeLayerMemory`` holds M_ont / M_fac / M_pas; this module re-exports the adapter for
any stale imports during the scaffold migration.
"""

from __future__ import annotations

from memgraphrag_core.memory.memgraph_memory import MemGraphGlobalMemory

# Backward-compatible name used by early scaffold code.
GlobalMemory = MemGraphGlobalMemory

__all__ = ["GlobalMemory", "MemGraphGlobalMemory"]

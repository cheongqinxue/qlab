"""Tests for deprecated ``GlobalMemory`` alias."""

from __future__ import annotations

from memgraphrag_core.global_memory import GlobalMemory, MemGraphGlobalMemory
from memgraphrag_core.memory import MemGraphGlobalMemory as ExportedMemory
from memgraphrag_core.memory import ThreeLayerMemory as ExportedThreeLayer


class TestGlobalMemoryAlias:
    """Backward-compatible imports."""

    def test_global_memory_is_subclass(self) -> None:
        assert GlobalMemory is MemGraphGlobalMemory

    def test_memory_package_exports(self) -> None:
        assert ExportedMemory is MemGraphGlobalMemory
        assert ExportedThreeLayer.__name__ == "ThreeLayerMemory"

"""Smoke imports for package ``__init__`` modules."""

from __future__ import annotations


class TestPackageInits:
    """Import side-effect-free package roots."""

    def test_memgraphrag_core_init(self) -> None:
        import memgraphrag_core

        assert "MemGraphRAG" in memgraphrag_core.__doc__ or memgraphrag_core.__doc__

    def test_subpackage_inits(self) -> None:
        import memgraphrag_core.agents
        import memgraphrag_core.indexing
        import memgraphrag_core.retrieval
        import memgraphrag_core.storage

        assert memgraphrag_core.agents.__doc__
        assert memgraphrag_core.indexing.__doc__
        assert memgraphrag_core.retrieval.__doc__
        assert memgraphrag_core.storage.__doc__

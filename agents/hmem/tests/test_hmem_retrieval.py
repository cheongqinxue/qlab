"""Retrieval pipeline tests (fixture mode)."""

from __future__ import annotations

import asyncio

from hmem_core.config import HMemConfig
from hmem_core.runtime import build_runtime
from hmem_core.toy_bootstrap import bootstrap_toy


class TestRetrievalPipeline:
    """End-to-end retrieval without fixture answer shortcut."""

    def test_toy_query_via_hybrid_retrieval(self) -> None:
        """Bootstrap toy corpus and answer via graph + tree ranking."""
        runtime = build_runtime(HMemConfig(), agents_mode="fixture")
        runtime.fixture_answer = None
        runtime.fixture_query = None
        asyncio.run(bootstrap_toy(runtime))
        runtime.fixture_answer = None
        runtime.refresh_engine()
        result = runtime.engine.ask("What is the cat's name?")
        assert result.answer == "Mochi"
        assert result.evidence_count >= 1

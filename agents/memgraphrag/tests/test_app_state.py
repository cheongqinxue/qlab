"""Tests for ``AppState``."""

from __future__ import annotations

from agents_utils.memory.hash_dense_encoder import HashDenseEncoder

from memgraphrag_core.agents.conflict_detect import FixtureConflictDetectionAgent
from memgraphrag_core.agents.conflict_resolve import FixtureConflictResolutionAgent
from memgraphrag_core.agents.extraction import FixtureExtractionAgent
from memgraphrag_core.indexing.orchestrator import IndexOrchestrator
from memgraphrag_core.memory.config import MemGraphMemoryConfig
from memgraphrag_core.memory.memgraph_memory import MemGraphGlobalMemory
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever
from memgraphrag_core.runtime import MemgraphRuntime
import importlib.util
from pathlib import Path as _Path


def _purge() -> None:
    _path = _Path(__file__).resolve().parent / "_path_setup.py"
    _spec = importlib.util.spec_from_file_location("mgr_path_setup", _path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    _mod.purge_shared_agent_modules()


class TestAppState:
    """Runtime dataclass."""

    def test_holds_memory(self) -> None:
        _purge()
        from server.app_state import AppState

        cfg = MemGraphMemoryConfig()
        encoder = HashDenseEncoder()
        memory = MemGraphGlobalMemory(cfg, encoder=encoder)
        orchestrator = IndexOrchestrator(
            memory,
            encoder,
            extraction=FixtureExtractionAgent({}),
            detection=FixtureConflictDetectionAgent([]),
            resolution=FixtureConflictResolutionAgent({}),
        )
        runtime = MemgraphRuntime(
            memory=memory,
            encoder=encoder,
            orchestrator=orchestrator,
            retriever=MemoryGuidedRetriever(memory, encoder, cfg),
        )
        state = AppState(runtime=runtime)
        assert state.memory is memory

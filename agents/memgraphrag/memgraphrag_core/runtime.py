"""Runtime wiring for MemGraphRAG server."""

from __future__ import annotations

import os
from dataclasses import dataclass

from hipporag_core.phrase_encoder import PhraseEncoder

from memgraphrag_core.agents.factory import build_indexing_agents
from memgraphrag_core.indexing.orchestrator import IndexOrchestrator
from memgraphrag_core.memory.config import MemGraphMemoryConfig
from memgraphrag_core.memory.memgraph_memory import MemGraphGlobalMemory
from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever
from memgraphrag_core.storage.neo4j_store import Neo4jMemoryProjection


@dataclass
class MemgraphRuntime:
    """Handles for memory, indexing, and retrieval."""

    memory: MemGraphGlobalMemory
    encoder: PhraseEncoder
    orchestrator: IndexOrchestrator
    retriever: MemoryGuidedRetriever
    neo4j: Neo4jMemoryProjection | None = None

    def close(self) -> None:
        """Release external resources."""
        if self.neo4j is not None:
            self.neo4j.close()


def create_encoder() -> PhraseEncoder:
    """Build phrase encoder (hash in tests, HF model otherwise)."""
    if os.environ.get("MEMGRAPHRAG_TEST_ENCODER") == "hash" or os.environ.get(
        "PYTEST_CURRENT_TEST"
    ):
        return PhraseEncoder()
    model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return PhraseEncoder.from_env_model(model)


def build_runtime(config: MemGraphMemoryConfig) -> MemgraphRuntime:
    """Construct runtime from config and environment."""
    encoder = create_encoder()
    memory = MemGraphGlobalMemory.build_or_restore(config, encoder=encoder)
    memory.set_encoder(encoder)

    agents = build_indexing_agents()
    neo4j: Neo4jMemoryProjection | None = None
    if config.neo4j_enabled and os.environ.get("MEMGRAPHRAG_NEO4J", "0") == "1":
        neo4j = Neo4jMemoryProjection.from_env()

    orchestrator = IndexOrchestrator(
        memory,
        encoder,
        extraction=agents[0],
        detection=agents[1],
        resolution=agents[2],
        neo4j=neo4j,
    )
    retriever = MemoryGuidedRetriever(memory, encoder, config)
    return MemgraphRuntime(
        memory=memory,
        encoder=encoder,
        orchestrator=orchestrator,
        retriever=retriever,
        neo4j=neo4j,
    )

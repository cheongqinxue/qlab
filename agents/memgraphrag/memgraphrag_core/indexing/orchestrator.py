"""Index orchestrator: passage ingest → A_ext → A_det → A_res → graph rebuild."""

from __future__ import annotations

from agents_utils.memory.base_encoder import BaseEncoder

from memgraphrag_core.agents.conflict_detect import FixtureConflictDetectionAgent
from memgraphrag_core.agents.conflict_resolve import FixtureConflictResolutionAgent
from memgraphrag_core.agents.extraction import FixtureExtractionAgent
from memgraphrag_core.agents.protocols import (
    ConflictDetectionAgent,
    ConflictResolutionAgent,
    ExtractionAgent,
)
from memgraphrag_core.indexing.merge import apply_extraction
from memgraphrag_core.memory.memgraph_memory import MemGraphGlobalMemory
from memgraphrag_core.storage.memory_store import save_memory
from memgraphrag_core.storage.neo4j_store import Neo4jMemoryProjection


class IndexOrchestrator:
    """Runs the multi-agent indexing loop over global memory."""

    def __init__(
        self,
        memory: MemGraphGlobalMemory,
        encoder: BaseEncoder,
        *,
        extraction: ExtractionAgent,
        detection: ConflictDetectionAgent,
        resolution: ConflictResolutionAgent,
        neo4j: Neo4jMemoryProjection | None = None,
    ) -> None:
        """Bind memory, agents, and optional Neo4j projection."""
        self._memory = memory
        self._encoder = encoder
        self._extraction = extraction
        self._detection = detection
        self._resolution = resolution
        self._neo4j = neo4j

    async def run_full_index(self) -> dict[str, int]:
        """Execute extraction, conflict handling, and graph projection."""
        new_fact_ids: list[str] = []
        for passage in self._memory.layers.iter_passages():
            payload = self._extraction.extract(passage.id, passage.text)
            new_fact_ids.extend(apply_extraction(self._memory.layers, passage.id, payload))

        conflict_groups = self._detection.detect(
            self._memory.layers,
            new_fact_ids=new_fact_ids,
        )
        for group in conflict_groups:
            self._resolution.resolve(self._memory.layers, group)

        self._memory.rebuild_index_graph(self._encoder)
        save_memory(self._memory.layers, self._memory.config.memory_data_dir)

        if self._neo4j is not None and self._memory.config.neo4j_enabled:
            self._neo4j.sync_layers(self._memory.layers, self._memory.index_graph)

        return {
            "passages": self._memory.passage_count,
            "ontology": self._memory.ontology_count,
            "facts": self._memory.fact_count,
            "conflicts_resolved": len(conflict_groups),
        }


def fixture_agents_from_eval(
    extraction_by_passage: dict[str, dict],
    conflict_groups: list[dict],
    resolutions_by_target: dict[str, dict],
) -> tuple[ExtractionAgent, ConflictDetectionAgent, ConflictResolutionAgent]:
    """Build fixture agent trio for toy eval corpora."""
    return (
        FixtureExtractionAgent(extraction_by_passage),
        FixtureConflictDetectionAgent(conflict_groups),
        FixtureConflictResolutionAgent(resolutions_by_target),
    )

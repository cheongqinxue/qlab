"""Agent role protocols for graph construction.

These are **not** LangGraph harness agents (cf. ReHarness). They are specialized
callables that maintain global memory consistency during offline indexing.
"""

from __future__ import annotations

from typing import Any, Protocol

from memgraphrag_core.memory.three_layer import ThreeLayerMemory


class ExtractionAgent(Protocol):
    """A_ext: chunk → candidate schemas, facts, and passage grounding (paper Eq. 3)."""

    def extract(self, passage_id: str, text: str) -> dict[str, Any]:
        """Return structured extraction deltas to merge into ``memory``.

        Args:
            passage_id: Stable chunk id.
            text: Passage body.

        Returns:
            Dict with keys such as ``schemas``, ``facts``, ``passage`` for the orchestrator.
        """


class ConflictDetectionAgent(Protocol):
    """A_det: scan M_fac for hard conflicts when new facts activate (paper Eq. 5)."""

    def detect(self, memory: ThreeLayerMemory, *, new_fact_ids: list[str]) -> list[dict[str, Any]]:
        """Return conflict groups with target fact ids and related fact ids.

        Args:
            memory: Current global memory.
            new_fact_ids: Recently activated fact ids to check against the corpus.

        Returns:
            Serialized conflict records (upstream ``detect_memory_conflicts`` shape).
        """


class ConflictResolutionAgent(Protocol):
    """A_res: adjudicate using M_pas evidence; filter, merge, or keep facts."""

    def resolve(
        self,
        memory: ThreeLayerMemory,
        conflict_group: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply resolution actions to memory (drop fact ids, merge entities, etc.).

        Args:
            memory: Global memory (mutated in place or via returned patch).
            conflict_group: One output from ``ConflictDetectionAgent``.

        Returns:
            Summary of actions taken (for Langfuse / API debug).
        """

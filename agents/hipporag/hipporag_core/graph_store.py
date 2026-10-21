"""Knowledge graph backing store (stub for Phase 1)."""

from __future__ import annotations

from hipporag_core.models import Passage, Triple


class GraphStore:
    """In-memory placeholder until persistent graph + PPR land."""

    def __init__(self) -> None:
        """Initialize empty passage and triple lists."""
        self._passages: dict[str, Passage] = {}
        self._triples: list[Triple] = []

    def add_passage(self, passage: Passage) -> None:
        """Register a passage for future indexing."""
        self._passages[passage.id] = passage

    def add_triple(self, triple: Triple) -> None:
        """Register an extracted triple."""
        self._triples.append(triple)

    @property
    def passage_count(self) -> int:
        """Number of indexed passages."""
        return len(self._passages)

    @property
    def triple_count(self) -> int:
        """Number of stored triples."""
        return len(self._triples)

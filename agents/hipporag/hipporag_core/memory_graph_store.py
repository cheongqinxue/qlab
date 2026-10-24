"""In-memory knowledge graph (tests and memory backend)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hipporag_core.content_hash import passage_content_hash
from hipporag_core.entity_ids import entity_id, normalize_phrase
from hipporag_core.models import Passage, Triple
from agents_utils.memory.base_encoder import BaseEncoder
from hipporag_core.ppr_snapshot import PprSnapshot


@dataclass
class _StagedPassage:
    passage: Passage
    content_hash: str
    indexed_hash: str | None = None


@dataclass
class MemoryGraphStore:
    """Entity–passage graph held in RAM."""

    _passages: dict[str, _StagedPassage] = field(default_factory=dict)
    _entities: dict[str, str] = field(default_factory=dict)
    _entity_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    _triples: list[Triple] = field(default_factory=list)
    _rel_edges: set[tuple[str, str]] = field(default_factory=set)
    _synonym_edges: set[tuple[str, str]] = field(default_factory=set)
    _mentions: dict[str, dict[str, int]] = field(default_factory=dict)

    def stage_passage(self, passage: Passage) -> None:
        """Register or update a passage awaiting indexing."""
        digest = passage_content_hash(passage.text)
        existing = self._passages.get(passage.id)
        if existing is None:
            self._passages[passage.id] = _StagedPassage(
                passage=passage,
                content_hash=digest,
            )
            return
        existing.passage = passage
        existing.content_hash = digest

    def pending_passage_ids(self) -> list[str]:
        """Passage ids whose content hash differs from last index."""
        pending: list[str] = []
        for pid, staged in self._passages.items():
            if staged.indexed_hash != staged.content_hash:
                pending.append(pid)
        return pending

    def mark_passage_indexed(self, passage_id: str) -> None:
        """Record successful index for idempotent skips."""
        staged = self._passages[passage_id]
        staged.indexed_hash = staged.content_hash

    def merge_entity(self, phrase: str, encoder: BaseEncoder) -> str:
        """Ensure entity node exists; return entity id."""
        text = normalize_phrase(phrase)
        eid = entity_id(text)
        self._entities[eid] = text
        if eid not in self._entity_vectors:
            self._entity_vectors[eid] = encoder.encode_document_numpy(text)
        return eid

    def add_mention(self, entity_eid: str, passage_id: str, count: int = 1) -> None:
        """Increment entity → passage mention count."""
        by_passage = self._mentions.setdefault(entity_eid, {})
        by_passage[passage_id] = by_passage.get(passage_id, 0) + count

    def add_rel(self, subject_eid: str, object_eid: str) -> None:
        """Add undirected relational edge between entities."""
        if subject_eid == object_eid:
            return
        pair = tuple(sorted((subject_eid, object_eid)))
        self._rel_edges.add(pair)

    def add_triple(self, triple: Triple) -> None:
        """Store triple and relational edge."""
        self._triples.append(triple)

    def add_synonym(self, eid_a: str, eid_b: str) -> None:
        """Add undirected synonymy edge."""
        if eid_a == eid_b:
            return
        pair = tuple(sorted((eid_a, eid_b)))
        self._synonym_edges.add(pair)

    def clear(self) -> None:
        """Drop all indexed graph content."""
        for staged in self._passages.values():
            staged.indexed_hash = None
        self._entities.clear()
        self._entity_vectors.clear()
        self._triples.clear()
        self._rel_edges.clear()
        self._synonym_edges.clear()
        self._mentions.clear()

    @property
    def passage_count(self) -> int:
        """Number of staged passages."""
        return len(self._passages)

    @property
    def triple_count(self) -> int:
        """Number of stored triples."""
        return len(self._triples)

    @property
    def entity_count(self) -> int:
        """Number of entity nodes."""
        return len(self._entities)

    @property
    def rel_edge_count(self) -> int:
        """Number of unique REL edges."""
        return len(self._rel_edges)

    @property
    def synonym_edge_count(self) -> int:
        """Number of synonymy edges."""
        return len(self._synonym_edges)

    def get_passage_text(self, passage_id: str) -> str:
        """Return passage body."""
        return self._passages[passage_id].passage.text

    def export_ppr_snapshot(self) -> PprSnapshot:
        """Build matrices for PPR and passage scoring."""
        entity_ids = sorted(self._entities.keys(), key=lambda e: self._entities[e])
        passage_ids = sorted(self._passages.keys())
        eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
        pid_to_idx = {pid: j for j, pid in enumerate(passage_ids)}

        entity_texts = [self._entities[eid] for eid in entity_ids]
        entity_vectors = np.stack(
            [self._entity_vectors[eid] for eid in entity_ids],
            axis=0,
        )

        edges: list[tuple[int, int, float]] = []
        for a, b in self._rel_edges | self._synonym_edges:
            edges.append((eid_to_idx[a], eid_to_idx[b], 1.0))

        mentions = np.zeros((len(entity_ids), len(passage_ids)), dtype=np.float64)
        for eid, by_passage in self._mentions.items():
            if eid not in eid_to_idx:
                continue
            row = eid_to_idx[eid]
            for pid, count in by_passage.items():
                if pid in pid_to_idx:
                    mentions[row, pid_to_idx[pid]] = float(count)

        return PprSnapshot(
            entity_ids=entity_ids,
            entity_texts=entity_texts,
            entity_vectors=entity_vectors,
            passage_ids=passage_ids,
            edges=edges,
            mentions=mentions,
        )

    def build_reset_weights(
        self,
        entity_indices: list[int],
        *,
        node_specificity: bool = True,
    ) -> np.ndarray:
        """Personalization vector for PPR (uniform on seeds × specificity).

        Args:
            entity_indices: Linked query entity node indices.
            node_specificity: If True, weight ``1 / (#passages mentioning node)``.

        Returns:
            Reset weights length ``|N|``.
        """
        snapshot = self.export_ppr_snapshot()
        reset = np.zeros(len(snapshot.entity_ids), dtype=np.float64)
        passage_counts = (snapshot.mentions > 0).sum(axis=1)
        for idx in entity_indices:
            weight = 1.0
            if node_specificity:
                count = int(passage_counts[idx])
                weight = 1.0 if count == 0 else 1.0 / count
            reset[idx] += weight
        return reset

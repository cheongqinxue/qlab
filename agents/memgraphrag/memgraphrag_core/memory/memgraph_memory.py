"""``BaseMemory`` implementation backed by three-layer global memory."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from agents_utils.memory.base_memory import BaseMemory, BaseMemoryItem
from hipporag_core.phrase_encoder import PhraseEncoder
from typing_extensions import Self

from memgraphrag_core.indexing.graph_builder import IndexingGraph, build_from_memory
from memgraphrag_core.memory.config import MemGraphMemoryConfig
from memgraphrag_core.memory.three_layer import ThreeLayerMemory
from memgraphrag_core.models import PassageRecord
from memgraphrag_core.storage.memory_store import load_memory, memory_json_path


class MemGraphGlobalMemory(BaseMemory):
    """Primary MemGraphRAG memory bank: three layers + graph-derived retrieval."""

    def __init__(
        self,
        config: MemGraphMemoryConfig,
        *,
        encoder: PhraseEncoder | None = None,
    ) -> None:
        """Attach configuration and empty layer store."""
        self._config = config
        self.layers = ThreeLayerMemory()
        self._encoder = encoder
        self.index_graph: IndexingGraph | None = None

    @property
    def config(self) -> MemGraphMemoryConfig:
        """Memory configuration."""
        return self._config

    @property
    def ontology_count(self) -> int:
        """Pass-through for health checks."""
        return self.layers.ontology_count

    @property
    def fact_count(self) -> int:
        """Pass-through for health checks."""
        return self.layers.fact_count

    @property
    def passage_count(self) -> int:
        """Pass-through for health checks."""
        return self.layers.passage_count

    def set_encoder(self, encoder: PhraseEncoder) -> None:
        """Attach phrase encoder used for graph vectors and retrieval."""
        self._encoder = encoder

    def rebuild_index_graph(self, encoder: PhraseEncoder | None = None) -> None:
        """Rebuild derived indexing graph from current layers."""
        enc = encoder or self._encoder
        if enc is None:
            enc = PhraseEncoder()
            self._encoder = enc
        if self.layers.passage_count == 0 and self.layers.fact_count == 0:
            self.index_graph = None
            return
        self.index_graph = build_from_memory(self.layers, enc)

    @classmethod
    def build_or_restore(cls, config: MemGraphMemoryConfig, **kwargs: Any) -> Self:
        """Construct memory and load ``memory.json`` when present."""
        encoder = kwargs.get("encoder")
        if encoder is not None and not isinstance(encoder, PhraseEncoder):
            msg = "encoder kwarg must be PhraseEncoder"
            raise TypeError(msg)
        inst = cls(config, encoder=encoder)
        data_dir = config.memory_data_dir
        if memory_json_path(data_dir).is_file():
            inst.layers = load_memory(data_dir)
            inst.rebuild_index_graph()
        return inst

    async def add(
        self,
        content: Any,
        *,
        vector: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Stage a passage in M_pas (index via ``POST /index``)."""
        _ = vector
        meta = metadata or {}
        if isinstance(content, PassageRecord):
            record = content
        elif isinstance(content, dict):
            record = PassageRecord(
                id=str(content.get("id", meta.get("passage_id", uuid.uuid4().hex))),
                text=str(content["text"]),
            )
        elif isinstance(content, str):
            passage_id = str(meta.get("passage_id", uuid.uuid4().hex))
            record = PassageRecord(id=passage_id, text=content)
        else:
            msg = "content must be PassageRecord, dict with text, or str"
            raise TypeError(msg)
        self.layers.add_passage(record)
        return record.id

    async def afind(
        self,
        *,
        query: str | None = None,
        vector: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[BaseMemoryItem]:
        """Memory-guided retrieval over indexed graph."""
        _ = vector
        if query is None or not query.strip():
            return []
        from memgraphrag_core.retrieval.retriever import MemoryGuidedRetriever

        top_k = int((metadata or {}).get("top_k", self._config.retrieval_top_k))
        encoder = self._encoder or PhraseEncoder()
        self._encoder = encoder
        retriever = MemoryGuidedRetriever(self, encoder, self._config)
        return await retriever.retrieve(query.strip(), top_k=top_k)

    async def adelete(self, item_id: str) -> bool:
        """Remove a fact or passage by id."""
        if self.layers.delete_fact(item_id):
            if self._encoder is not None:
                self.rebuild_index_graph()
            return True
        if self.layers.delete_passage(item_id):
            return True
        return False

    async def aclear(self) -> None:
        """Clear all layers and drop derived graph snapshots."""
        self.layers.clear()
        self.index_graph = None

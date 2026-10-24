"""Memory-aware retriever orchestrating filter → seeds → PPR → passage ranking."""

from __future__ import annotations

from agents_utils.memory.base_memory import BaseMemoryItem
from agents_utils.graph.ppr import run_personalized_pagerank
from agents_utils.memory.base_encoder import BaseEncoder

from memgraphrag_core.indexing.graph_builder import IndexingGraph
from memgraphrag_core.memory.config import MemGraphMemoryConfig
from memgraphrag_core.memory.memgraph_memory import MemGraphGlobalMemory
from memgraphrag_core.retrieval.layer_filter import filter_layers_by_query
from memgraphrag_core.retrieval.ppr_seeds import build_ppr_seeds


class MemoryGuidedRetriever:
    """Query-time retrieval over indexed global memory + graph snapshot."""

    def __init__(
        self,
        memory: MemGraphGlobalMemory,
        encoder: BaseEncoder,
        config: MemGraphMemoryConfig,
    ) -> None:
        """Attach memory bank and encoder."""
        self._memory = memory
        self._encoder = encoder
        self._config = config

    async def retrieve(self, query: str, *, top_k: int | None = None) -> list[BaseMemoryItem]:
        """Run memory-guided retrieval for one query."""
        k = top_k if top_k is not None else self._config.retrieval_top_k
        graph = self._memory.index_graph
        if graph is None or graph.node_count == 0:
            return self._dense_passage_fallback(query, top_k=k)

        filtered = filter_layers_by_query(
            self._memory.layers,
            self._encoder,
            query,
            sim_threshold=self._config.retrieval_sim_threshold,
        )
        if not filtered:
            return self._dense_passage_fallback(query, top_k=k)

        key_to_index = graph.key_to_index()
        reset = build_ppr_seeds(
            filtered,
            num_nodes=graph.node_count,
            key_to_index=key_to_index,
        )
        scores = run_personalized_pagerank(
            graph.node_count,
            graph.edges,
            reset,
            damping=self._config.ppr_damping,
        )

        ranked: list[tuple[int, float]] = [
            (idx, float(scores[idx])) for idx in graph.passage_indices
        ]
        ranked.sort(key=lambda pair: pair[1], reverse=True)

        items: list[BaseMemoryItem] = []
        for idx, score in ranked[:k]:
            key = graph.node_keys[idx]
            passage = self._memory.layers.get_passage(key.record_id)
            if passage is None:
                continue
            items.append(
                BaseMemoryItem(
                    id=passage.id,
                    similarity=score,
                    content=passage.text,
                )
            )
        if items:
            return items
        return self._dense_passage_fallback(query, top_k=k)

    def _dense_passage_fallback(self, query: str, *, top_k: int) -> list[BaseMemoryItem]:
        """Rank passages by cosine similarity when graph/PPR is unavailable."""
        query_vec = self._encoder.encode_query_numpy(query)
        scored: list[tuple[str, float, str]] = []
        for passage in self._memory.layers.iter_passages():
            sim = float(query_vec @ self._encoder.encode_document_numpy(passage.text))
            scored.append((passage.id, sim, passage.text))
        scored.sort(key=lambda row: row[1], reverse=True)
        return [
            BaseMemoryItem(id=pid, similarity=sim, content=text)
            for pid, sim, text in scored[:top_k]
        ]

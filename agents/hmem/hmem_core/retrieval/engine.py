"""Online retrieval orchestrator (paper §3.3)."""

from __future__ import annotations

from datetime import datetime, timezone

from agents_utils.memory.base_encoder import BaseEncoder

from hmem_core.config import HMemConfig
from hmem_core.hybrid_store import InMemoryHybridStore
from hmem_core.models import AskResponse, Evidence
from hmem_core.retrieval.fixture_retrieval import fixture_answer
from hmem_core.retrieval.graph_search import GraphSearcher
from hmem_core.retrieval.planner import RetrievalPlanner, build_planner
from hmem_core.retrieval.scoring import (
    combined_score,
    memory_robustness,
    semantic_similarity,
    temporal_relevance,
)
from hmem_core.retrieval.synthesizer import AnswerSynthesizer, build_synthesizer
from hmem_core.retrieval.tree_search import TreeSearcher


class RetrievalEngine:
    """End-to-end ask pipeline."""

    def __init__(
        self,
        store: InMemoryHybridStore,
        config: HMemConfig,
        encoder: BaseEncoder,
        *,
        agents_mode: str,
        fixture_query: str | None = None,
        fixture_answer_text: str | None = None,
    ) -> None:
        """Wire sub-searchers and agents."""
        self._store = store
        self._config = config
        self._encoder = encoder
        self._agents_mode = agents_mode
        self._fixture_query = fixture_query
        self._fixture_answer_text = fixture_answer_text
        self._planner = RetrievalPlanner(build_planner(agents_mode=agents_mode))
        self._synthesizer = AnswerSynthesizer(build_synthesizer(agents_mode=agents_mode))
        self._graph_search = GraphSearcher(store.graph, config)
        tree = store.tree
        self._tree_search = TreeSearcher(tree) if tree is not None else None

    def ask(self, query: str) -> AskResponse:
        """Answer ``query`` using hybrid retrieval."""
        result = self._retrieve(query)
        if (
            not result.answer
            and self._fixture_answer_text
            and self._agents_mode == "fixture"
        ):
            return fixture_answer(
                self._store,
                query=query,
                fixture_query=self._fixture_query,
                fixture_answer_text=self._fixture_answer_text,
                agents_mode=self._agents_mode,
            )
        return result

    def _retrieve(self, query: str) -> AskResponse:
        """Plan → graph → tree → rank → synthesize."""
        sub_queries = self._planner.decompose(query)
        query_time = datetime.now(tz=timezone.utc)
        query_vec = self._encoder.encode_query_numpy(query)
        evidence: list[Evidence] = []
        for sub in sub_queries:
            seeds = self._graph_search.seed_entities_from_text(sub.text)
            seeds |= self._graph_search.expand_from_query_entities(list(seeds))
            fragment_ids = self._graph_search.fragment_ids_for_entities(seeds)
            allow = fragment_ids if fragment_ids else None
            if self._tree_search:
                evidence.extend(
                    self._tree_search.collect(
                        sub,
                        allowed_fragment_ids=allow,
                    )
                )
            elif fragment_ids:
                for fragment_id in fragment_ids:
                    fragment = self._store.fragments.get(fragment_id)
                    if fragment:
                        evidence.append(
                            Evidence(
                                text=fragment.text,
                                fragment_id=fragment_id,
                                start_time=fragment.timestamp,
                                end_time=fragment.timestamp,
                            )
                        )
        ranked = self._rank_evidence(query_vec, evidence, sub_queries[0], query_time)
        top = ranked[: self._config.retrieval_top_k]
        texts = [ev.text for ev in top]
        answer = self._synthesizer.answer(query, texts) if texts else ""
        return AskResponse(
            query=query,
            answer=answer,
            evidence_count=len(top),
            agents_mode=self._agents_mode,
        )

    def _rank_evidence(
        self,
        query_vec,
        evidence: list[Evidence],
        sub_query,
        query_time: datetime,
    ) -> list[Evidence]:
        """Apply paper ranking terms and sort descending."""
        ranked: list[Evidence] = []
        seen: set[str] = set()
        for item in evidence:
            key = item.text.strip()
            if key in seen:
                continue
            seen.add(key)
            mem_vec = self._encoder.encode_query_numpy(item.text)
            item.semantic = semantic_similarity(query_vec, mem_vec)
            q_tokens = {
                w.strip("?.,").lower()
                for w in sub_query.text.split()
                if len(w.strip("?.,")) > 3
            }
            text_lower = item.text.lower()
            overlap = sum(1 for w in q_tokens if w in text_lower)
            if overlap:
                item.semantic = min(1.0, item.semantic + 0.05 * overlap)
            start = item.start_time or query_time
            end = item.end_time or start
            item.temporal = temporal_relevance(
                start,
                end,
                sub_query.time_start,
                sub_query.time_end,
                lam=self._config.temporal_lambda,
            )
            item.robustness = memory_robustness(
                query_time,
                end,
                consolidation_count=item.consolidation_count,
                tau=self._config.robustness_tau_seconds,
                eta=self._config.robustness_eta,
            )
            item.score = combined_score(item, self._config)
            ranked.append(item)
        ranked.sort(key=lambda ev: ev.score, reverse=True)
        return ranked

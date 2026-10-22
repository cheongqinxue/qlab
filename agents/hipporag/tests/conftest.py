"""Pytest fixtures for HippoRAG unit tests (in-memory graph, no Neo4j required)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def hipporag_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid Neo4j / HuggingFace during unit tests."""
    monkeypatch.setenv("HIPPORAG_GRAPH_BACKEND", "memory")
    monkeypatch.setenv("HIPPORAG_TEST_ENCODER", "hash")
    monkeypatch.setenv("HIPPORAG_BOOTSTRAP_TOY", "0")

from hipporag_core.memory_graph_store import MemoryGraphStore
from hipporag_core.models import OpenIEExtraction, Passage, Triple
from hipporag_core.phrase_encoder import PhraseEncoder


@pytest.fixture
def phrase_encoder() -> PhraseEncoder:
    """Deterministic tiny encoder for linking/synonymy tests (no HF download)."""

    return PhraseEncoder.from_vectors(
        {
            "stanford": _unit(0),
            "professor thomas": _unit(1),
            "alzheimer's disease": _unit(2),
            "harvard": _unit(3),
            "stanford university": _unit(0, jitter=0.02),
            "alzheimers": _unit(2, jitter=0.03),
        }
    )


def _unit(dim: int, *, jitter: float = 0.0) -> np.ndarray:
    vec = np.zeros(8, dtype=np.float64)
    vec[dim % 8] = 1.0
    if jitter:
        vec[(dim + 1) % 8] = jitter
    return vec / np.linalg.norm(vec)


@pytest.fixture
def toy_passages() -> list[Passage]:
    """Mini multi-hop corpus (paper Figure-1 style)."""

    return [
        Passage(
            id="p_stanford",
            text="Stanford University is located in California.",
        ),
        Passage(
            id="p_prof_stanford",
            text="Professor Thomas is a faculty member at Stanford.",
        ),
        Passage(
            id="p_prof_alz",
            text="Professor Thomas researches the neuroscience of Alzheimer's disease.",
        ),
        Passage(
            id="p_alz",
            text="Alzheimer's disease is a neurodegenerative disorder affecting memory.",
        ),
        Passage(
            id="p_harvard",
            text="Harvard University is an Ivy League school in Massachusetts.",
        ),
    ]


@pytest.fixture
def toy_openie_by_passage() -> dict[str, OpenIEExtraction]:
    """Fixed OpenIE outputs for the toy corpus (no LLM)."""

    return {
        "p_stanford": OpenIEExtraction(
            passage_id="p_stanford",
            entities=["Stanford University", "California"],
            triples=[
                Triple(
                    subject="Stanford University",
                    predicate="is located in",
                    object="California",
                    passage_id="p_stanford",
                ),
            ],
        ),
        "p_prof_stanford": OpenIEExtraction(
            passage_id="p_prof_stanford",
            entities=["Professor Thomas", "Stanford"],
            triples=[
                Triple(
                    subject="Professor Thomas",
                    predicate="works at",
                    object="Stanford",
                    passage_id="p_prof_stanford",
                ),
            ],
        ),
        "p_prof_alz": OpenIEExtraction(
            passage_id="p_prof_alz",
            entities=["Professor Thomas", "Alzheimer's disease"],
            triples=[
                Triple(
                    subject="Professor Thomas",
                    predicate="researches",
                    object="Alzheimer's disease",
                    passage_id="p_prof_alz",
                ),
            ],
        ),
        "p_alz": OpenIEExtraction(
            passage_id="p_alz",
            entities=["Alzheimer's disease", "memory"],
            triples=[
                Triple(
                    subject="Alzheimer's disease",
                    predicate="affects",
                    object="memory",
                    passage_id="p_alz",
                ),
            ],
        ),
        "p_harvard": OpenIEExtraction(
            passage_id="p_harvard",
            entities=["Harvard University", "Massachusetts"],
            triples=[
                Triple(
                    subject="Harvard University",
                    predicate="is in",
                    object="Massachusetts",
                    passage_id="p_harvard",
                ),
            ],
        ),
    }


@pytest.fixture
def indexed_toy_graph(
    toy_passages: list[Passage],
    toy_openie_by_passage: dict[str, OpenIEExtraction],
    phrase_encoder: PhraseEncoder,
) -> MemoryGraphStore:
    """Toy corpus fully indexed in memory."""

    from hipporag_core.indexer import Indexer

    store = MemoryGraphStore()
    indexer = Indexer(store=store, encoder=phrase_encoder, synonymy_threshold=0.8)
    for passage in toy_passages:
        store.stage_passage(passage)
    indexer.index_passages(openie_by_passage=toy_openie_by_passage)
    indexer.add_synonymy_edges()
    return store

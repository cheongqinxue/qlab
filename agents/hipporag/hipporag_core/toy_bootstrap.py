"""Load and index the toy multi-hop corpus (no LLM OpenIE)."""

from __future__ import annotations

import os

from hipporag_core.eval_loader import load_toy_corpus, load_toy_openie_fixtures, toy_multihop_dir
from hipporag_core.models import OpenIEExtraction
from hipporag_core.runtime import HipporagRuntime


def bootstrap_toy_enabled() -> bool:
    """Whether startup should load toy corpus into the graph."""
    return os.environ.get("HIPPORAG_BOOTSTRAP_TOY", "1") != "0"


def load_toy_passages(runtime: HipporagRuntime) -> int:
    """Stage passages from ``eval/toy_multihop/corpus.json``.

    Args:
        runtime: Active HippoRAG runtime.

    Returns:
        Number of passages staged.
    """
    for passage in load_toy_corpus():
        runtime.store.stage_passage(passage)
    return runtime.store.passage_count


def index_toy_graph(runtime: HipporagRuntime) -> dict[str, object]:
    """Index staged toy passages using fixture OpenIE (Neo4j + in-memory graph).

    Args:
        runtime: Active HippoRAG runtime.

    Returns:
        Summary counts from indexing.
    """
    openie_map: dict[str, OpenIEExtraction] = load_toy_openie_fixtures(toy_multihop_dir())
    indexed = runtime.indexer.index_passages(openie_by_passage=openie_map)
    synonyms = runtime.indexer.add_synonymy_edges()
    store = runtime.store
    return {
        "indexed_passages": indexed,
        "synonym_edges_added": synonyms,
        "passages": store.passage_count,
        "entities": store.entity_count,
        "triples": store.triple_count,
        "rel_edges": store.rel_edge_count,
        "synonym_edges": store.synonym_edge_count,
    }


def bootstrap_toy_multihop(runtime: HipporagRuntime) -> dict[str, object] | None:
    """Load toy corpus and index if bootstrap is enabled.

    Args:
        runtime: Active HippoRAG runtime.

    Returns:
        Index summary when bootstrap ran, else ``None``.
    """
    if not bootstrap_toy_enabled():
        return None
    load_toy_passages(runtime)
    return index_toy_graph(runtime)

"""Load toy eval JSON into the hybrid store."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from hmem_core.indexing.fixture_indexer import index_fragments, run_consolidation_pass
from hmem_core.models import MemoryFragment
from hmem_core.runtime import HMemRuntime

_TOY_DIR = Path(__file__).resolve().parents[1] / "eval" / "toy_locomo_mini"
_TOY_JSON = _TOY_DIR / "fragments.json"


def load_toy_bundle() -> dict:
    """Read toy fragments, graph edges, and fixture QA."""
    return json.loads(_TOY_JSON.read_text(encoding="utf-8"))


async def bootstrap_toy(runtime: HMemRuntime) -> dict[str, object]:
    """Index toy corpus and run consolidation."""
    bundle = load_toy_bundle()
    runtime.store.clear()
    runtime.store.config = runtime.config
    fragments = [
        MemoryFragment(
            id=item["id"],
            text=item["text"],
            timestamp=datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
            speaker=item.get("speaker"),
        )
        for item in bundle["fragments"]
    ]
    from hmem_core.models import GraphEdge

    edges = [GraphEdge.model_validate(e) for e in bundle.get("graph_edges", [])]
    leaves = index_fragments(
        runtime.store,
        fragments,
        edges,
        config=runtime.config,
        encoder=runtime.encoder,
        agents_mode=runtime.agents_mode,
    )
    summaries = run_consolidation_pass(
        runtime.store,
        config=runtime.config,
        encoder=runtime.encoder,
        agents_mode=runtime.agents_mode,
    )
    qa = bundle.get("fixture_qa", {})
    runtime.fixture_query = qa.get("query")
    runtime.fixture_answer = qa.get("answer")
    runtime.refresh_engine()
    return {
        "source": "toy_locomo_mini",
        "fragments": leaves,
        "entities": runtime.store.entity_count,
        "summary_nodes": summaries,
        "extraction": runtime.agents_mode,
    }

"""Bootstrap toy conflict corpus with fixture agents."""

from __future__ import annotations

import os

from memgraphrag_core.eval_loader import load_toy_corpus
from memgraphrag_core.models import PassageRecord
from memgraphrag_core.runtime import MemgraphRuntime


def bootstrap_toy_enabled() -> bool:
    """Whether startup should load and index toy corpus."""
    return os.environ.get("MEMGRAPHRAG_BOOTSTRAP_TOY", "0") == "1"


async def load_toy_passages(runtime: MemgraphRuntime) -> int:
    """Stage ``eval/toy_conflict`` passages into M_pas."""
    for row in load_toy_corpus():
        await runtime.memory.add({"id": row["id"], "text": row["text"]})
    return runtime.memory.passage_count


async def index_toy_memory(runtime: MemgraphRuntime) -> dict[str, int]:
    """Run full fixture index pipeline on staged toy corpus."""
    return await runtime.orchestrator.run_full_index()


async def bootstrap_toy_conflict(runtime: MemgraphRuntime) -> dict[str, int] | None:
    """Load and index toy data when enabled."""
    if not bootstrap_toy_enabled():
        return None
    await load_toy_passages(runtime)
    return await index_toy_memory(runtime)

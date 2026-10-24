"""Factory for indexing agents (fixture vs LLM)."""

from __future__ import annotations

import os

from memgraphrag_core.agents.conflict_detect import (
    FixtureConflictDetectionAgent,
    LlmConflictDetectionAgent,
)
from memgraphrag_core.agents.conflict_resolve import (
    FixtureConflictResolutionAgent,
    LlmConflictResolutionAgent,
)
from memgraphrag_core.agents.extraction import FixtureExtractionAgent, LlmExtractionAgent
from memgraphrag_core.agents.protocols import (
    ConflictDetectionAgent,
    ConflictResolutionAgent,
    ExtractionAgent,
)
from memgraphrag_core.eval_loader import load_toy_conflicts, load_toy_extraction, load_toy_resolutions
from memgraphrag_core.indexing.orchestrator import fixture_agents_from_eval


def agents_use_fixture() -> bool:
    """True when indexing agents should use committed JSON fixtures."""
    mode = os.environ.get("MEMGRAPHRAG_AGENTS", "fixture").strip().lower()
    if mode == "llm":
        return False
    if mode == "fixture":
        return True
    return os.environ.get("MEMGRAPHRAG_AGENTS_FIXTURE", "1") != "0"


def build_indexing_agents() -> tuple[ExtractionAgent, ConflictDetectionAgent, ConflictResolutionAgent]:
    """Create A_ext, A_det, A_res from environment."""
    if agents_use_fixture():
        return fixture_agents_from_eval(
            load_toy_extraction(),
            load_toy_conflicts(),
            load_toy_resolutions(),
        )
    return (
        LlmExtractionAgent(),
        LlmConflictDetectionAgent(),
        LlmConflictResolutionAgent(),
    )

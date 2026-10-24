"""Factory for CoVe step agents (fixture vs LLM)."""

from __future__ import annotations

import os

from cove_core.agents.fixture import (
    FixtureBaselineDrafter,
    FixtureFinalReviser,
    FixtureVerificationExecutor,
    FixtureVerificationPlanner,
)
from cove_core.agents.llm_stub import (
    LlmBaselineDrafter,
    LlmFinalReviser,
    LlmVerificationExecutor,
    LlmVerificationPlanner,
)
from cove_core.protocols import (
    BaselineDrafter,
    FinalReviser,
    VerificationExecutor,
    VerificationPlanner,
)


def agents_use_fixture() -> bool:
    """True when CoVe should use committed eval fixtures."""
    mode = os.environ.get("COVE_AGENTS", "fixture").strip().lower()
    if mode == "llm":
        return False
    if mode == "fixture":
        return True
    return os.environ.get("COVE_AGENTS_FIXTURE", "1") != "0"


def build_cove_agents() -> tuple[
    BaselineDrafter,
    VerificationPlanner,
    VerificationExecutor,
    FinalReviser,
]:
    """Create the four CoVe role implementations from environment."""
    if agents_use_fixture():
        return (
            FixtureBaselineDrafter(),
            FixtureVerificationPlanner(),
            FixtureVerificationExecutor(),
            FixtureFinalReviser(),
        )
    return (
        LlmBaselineDrafter(),
        LlmVerificationPlanner(),
        LlmVerificationExecutor(),
        LlmFinalReviser(),
    )

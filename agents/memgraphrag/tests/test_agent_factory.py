"""Tests for agent factory."""

from __future__ import annotations

import pytest

from memgraphrag_core.agents.conflict_detect import FixtureConflictDetectionAgent, LlmConflictDetectionAgent
from memgraphrag_core.agents.extraction import FixtureExtractionAgent, LlmExtractionAgent
from memgraphrag_core.agents.factory import agents_use_fixture, build_indexing_agents


class TestAgentFactory:
    """Environment-driven agent selection."""

    def test_defaults_to_fixture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MEMGRAPHRAG_AGENTS", raising=False)
        monkeypatch.setenv("MEMGRAPHRAG_AGENTS_FIXTURE", "1")
        assert agents_use_fixture() is True

    def test_llm_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMGRAPHRAG_AGENTS", "llm")
        assert agents_use_fixture() is False

    def test_build_fixture_agents(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMGRAPHRAG_AGENTS", "fixture")
        ext, det, _res = build_indexing_agents()
        assert isinstance(ext, FixtureExtractionAgent)
        assert isinstance(det, FixtureConflictDetectionAgent)

    def test_build_llm_agents(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMGRAPHRAG_AGENTS", "llm")
        ext, det, res = build_indexing_agents()
        assert isinstance(ext, LlmExtractionAgent)
        assert isinstance(det, LlmConflictDetectionAgent)
        from memgraphrag_core.agents.conflict_resolve import LlmConflictResolutionAgent

        assert isinstance(res, LlmConflictResolutionAgent)

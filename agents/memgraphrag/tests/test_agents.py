"""Tests for extraction and conflict agents."""

from __future__ import annotations

from unittest.mock import MagicMock

from memgraphrag_core.agents.conflict_detect import (
    FixtureConflictDetectionAgent,
    LlmConflictDetectionAgent,
)
from memgraphrag_core.agents.conflict_resolve import (
    FixtureConflictResolutionAgent,
    LlmConflictResolutionAgent,
)
from memgraphrag_core.agents.extraction import FixtureExtractionAgent, LlmExtractionAgent
from memgraphrag_core.agents.llm_models import (
    ConflictDetectionResult,
    ConflictGroupItem,
    ExtractionResult,
    FactItem,
    ResolutionResult,
    SchemaItem,
)
from memgraphrag_core.memory.three_layer import ThreeLayerMemory


class TestExtractionAgents:
    """A_ext implementations."""

    def test_fixture_returns_map_entry(self, toy_extraction_by_passage: dict) -> None:
        agent = FixtureExtractionAgent(toy_extraction_by_passage)
        out = agent.extract("p1", "ignored text")
        assert out["facts"][0]["id"] == "f1"

    def test_fixture_missing_passage_returns_empty(self) -> None:
        agent = FixtureExtractionAgent({})
        out = agent.extract("missing", "text")
        assert out == {"schemas": [], "facts": []}

    def test_llm_extract_structured(self) -> None:
        runner = MagicMock()
        runner.invoke.return_value = ExtractionResult(
            schemas=[SchemaItem(id="s1", label="BirthYear")],
            facts=[FactItem(id="f1", text="Born 1642", passage_ids=[], schema_ids=["s1"])],
        )
        agent = LlmExtractionAgent(runner=runner)
        out = agent.extract("p1", "Newton was born in 1642.")
        assert out["facts"][0]["passage_ids"] == ["p1"]
        assert runner.invoke.called


class TestConflictAgents:
    """A_det and A_res implementations."""

    def test_fixture_detect_filters_by_target(self) -> None:
        groups = [{"target_id": "f_a", "related_ids": ["f_b"]}]
        agent = FixtureConflictDetectionAgent(groups)
        found = agent.detect(ThreeLayerMemory(), new_fact_ids=["f_a", "f_c"])
        assert len(found) == 1
        assert found[0]["target_id"] == "f_a"

    def test_fixture_detect_no_match(self) -> None:
        agent = FixtureConflictDetectionAgent([{"target_id": "f_x"}])
        assert agent.detect(ThreeLayerMemory(), new_fact_ids=["f_y"]) == []

    def test_llm_detect_filters_to_new_facts(self) -> None:
        runner = MagicMock()
        runner.invoke.return_value = ConflictDetectionResult(
            groups=[
                ConflictGroupItem(
                    target_id="f_new",
                    related_ids=["f_old"],
                    description="conflict",
                )
            ]
        )
        agent = LlmConflictDetectionAgent(runner=runner)
        found = agent.detect(ThreeLayerMemory(), new_fact_ids=["f_new"])
        assert len(found) == 1
        assert found[0]["target_id"] == "f_new"

    def test_llm_detect_empty_when_no_new_facts(self) -> None:
        agent = LlmConflictDetectionAgent(runner=MagicMock())
        assert agent.detect(ThreeLayerMemory(), new_fact_ids=[]) == []

    def test_fixture_resolve_drops_facts(self, three_layer_with_newton_facts) -> None:
        memory = three_layer_with_newton_facts
        assert memory.fact_count == 2
        agent = FixtureConflictResolutionAgent(
            {"f_newton_1643": {"drop": ["f_newton_1643"], "keep": ["f_newton_1642"]}}
        )
        summary = agent.resolve(memory, {"target_id": "f_newton_1643"})
        assert summary["dropped_fact_ids"] == ["f_newton_1643"]
        assert memory.fact_count == 1

    def test_fixture_resolve_unknown_target_no_drops(self, three_layer_with_newton_facts) -> None:
        memory = three_layer_with_newton_facts
        agent = FixtureConflictResolutionAgent({})
        summary = agent.resolve(memory, {"target_id": "missing"})
        assert summary["dropped_fact_ids"] == []

    def test_llm_resolve_applies_drop(self, three_layer_with_newton_facts) -> None:
        memory = three_layer_with_newton_facts
        runner = MagicMock()
        runner.invoke.return_value = ResolutionResult(
            drop=["f_newton_1643"],
            keep=["f_newton_1642"],
            rationale="fixture test",
        )
        agent = LlmConflictResolutionAgent(runner=runner)
        summary = agent.resolve(memory, {"target_id": "f_newton_1643", "related_ids": []})
        assert "f_newton_1643" in summary["dropped_fact_ids"]
        assert memory.fact_count == 1

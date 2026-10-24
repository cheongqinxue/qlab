"""Conflict detection agent (A_det) implementations."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from memgraphrag_core.agents.llm_client import create_chat_model
from memgraphrag_core.agents.llm_models import ConflictDetectionResult
from memgraphrag_core.agents.prompts import CONFLICT_DETECTION_HUMAN, CONFLICT_DETECTION_SYSTEM
from memgraphrag_core.memory.three_layer import ThreeLayerMemory


def _facts_json(memory: ThreeLayerMemory) -> str:
    """Serialize M_fac for LLM context."""
    rows = [
        {
            "id": fact.id,
            "text": fact.text,
            "schema_ids": fact.schema_ids,
            "passage_ids": fact.passage_ids,
        }
        for fact in memory.iter_facts()
    ]
    return json.dumps(rows, indent=2)


class LlmConflictDetectionAgent:
    """LLM-based hard conflict detection."""

    def __init__(self, model: str | None = None, *, runner: object | None = None) -> None:
        """Bind structured LLM runner (injectable for tests)."""
        if runner is None:
            runner = create_chat_model(model).with_structured_output(ConflictDetectionResult)
        self._runner = runner

    def detect(self, memory: ThreeLayerMemory, *, new_fact_ids: list[str]) -> list[dict]:
        """Scan for conflicts involving ``new_fact_ids``."""
        if not new_fact_ids:
            return []
        result: ConflictDetectionResult = self._runner.invoke(
            [
                SystemMessage(content=CONFLICT_DETECTION_SYSTEM),
                HumanMessage(
                    content=CONFLICT_DETECTION_HUMAN.format(
                        new_fact_ids=json.dumps(new_fact_ids),
                        facts_json=_facts_json(memory),
                    ),
                ),
            ]
        )
        allowed = set(new_fact_ids)
        groups: list[dict] = []
        for group in result.groups:
            if group.target_id not in allowed:
                continue
            groups.append(group.model_dump())
        return groups


class FixtureConflictDetectionAgent:
    """Return pre-recorded conflict groups for toy eval."""

    def __init__(self, conflict_groups: list[dict]) -> None:
        """Store fixture conflict list."""
        self._groups = conflict_groups

    def detect(self, memory: ThreeLayerMemory, *, new_fact_ids: list[str]) -> list[dict]:
        """Filter fixture groups relevant to ``new_fact_ids``."""
        _ = memory
        return [g for g in self._groups if g.get("target_id") in new_fact_ids]

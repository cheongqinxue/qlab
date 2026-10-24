"""Conflict resolution agent (A_res) implementations."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from memgraphrag_core.agents.llm_client import create_chat_model
from memgraphrag_core.agents.llm_models import ResolutionResult
from memgraphrag_core.agents.prompts import CONFLICT_RESOLUTION_HUMAN, CONFLICT_RESOLUTION_SYSTEM
from memgraphrag_core.memory.three_layer import ThreeLayerMemory


def _passage_evidence_json(memory: ThreeLayerMemory, conflict_group: dict) -> str:
    """Collect passage texts linked to facts in the conflict group."""
    fact_ids = {str(conflict_group.get("target_id", ""))}
    fact_ids.update(str(x) for x in conflict_group.get("related_ids", []))
    passage_ids: set[str] = set()
    for fid in fact_ids:
        fact = memory.get_fact(fid)
        if fact is not None:
            passage_ids.update(fact.passage_ids)
    rows = []
    for pid in sorted(passage_ids):
        passage = memory.get_passage(pid)
        if passage is not None:
            rows.append({"id": pid, "text": passage.text})
    return json.dumps(rows, indent=2)


class LlmConflictResolutionAgent:
    """Evidence-grounded resolution via LLM."""

    def __init__(self, model: str | None = None, *, runner: object | None = None) -> None:
        """Bind structured LLM runner (injectable for tests)."""
        if runner is None:
            runner = create_chat_model(model).with_structured_output(ResolutionResult)
        self._runner = runner

    def resolve(self, memory: ThreeLayerMemory, conflict_group: dict) -> dict:
        """Adjudicate one conflict group and apply drops to M_fac."""
        result: ResolutionResult = self._runner.invoke(
            [
                SystemMessage(content=CONFLICT_RESOLUTION_SYSTEM),
                HumanMessage(
                    content=CONFLICT_RESOLUTION_HUMAN.format(
                        group_json=json.dumps(conflict_group, indent=2),
                        passages_json=_passage_evidence_json(memory, conflict_group),
                    ),
                ),
            ]
        )
        dropped: list[str] = []
        for fid in result.drop:
            if memory.delete_fact(str(fid)):
                dropped.append(str(fid))
        return {
            "target_id": str(conflict_group.get("target_id", "")),
            "dropped_fact_ids": dropped,
            "kept_fact_ids": list(result.keep),
            "rationale": result.rationale,
        }


class FixtureConflictResolutionAgent:
    """Apply predetermined keep/drop fact id lists."""

    def __init__(self, resolutions_by_target: dict[str, dict]) -> None:
        """Map conflict target fact id → ``{drop: [...], keep: [...]}``."""
        self._resolutions = resolutions_by_target

    def resolve(self, memory: ThreeLayerMemory, conflict_group: dict) -> dict:
        """Remove dropped facts from M_fac."""
        target = str(conflict_group.get("target_id", ""))
        plan = self._resolutions.get(target, {})
        dropped: list[str] = []
        for fid in plan.get("drop", []):
            if memory.delete_fact(str(fid)):
                dropped.append(str(fid))
        return {"target_id": target, "dropped_fact_ids": dropped}

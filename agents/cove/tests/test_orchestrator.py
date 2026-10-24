"""CoVe orchestrator with fixture agents."""

from __future__ import annotations

import asyncio

from cove_core.agents.factory import build_cove_agents
from cove_core.config import CoVeConfig
from cove_core.eval_loader import load_toy_manifest
from cove_core.orchestrator import CoVeOrchestrator


class TestCoVeOrchestrator:
    """End-to-end fixture pipeline."""

    def test_mexican_american_war_fixes_end_year(self) -> None:
        manifest = load_toy_manifest()
        query_id = str(manifest["query_id"])
        query = str(manifest["query"])
        drafter, planner, executor, reviser = build_cove_agents()
        orchestrator = CoVeOrchestrator(
            CoVeConfig(),
            drafter=drafter,
            planner=planner,
            executor=executor,
            reviser=reviser,
        )
        result = asyncio.run(orchestrator.run(query, query_id=query_id))
        assert "1849" in result.baseline_response
        assert "1848" in result.final_response
        assert len(result.verification_questions) == 2
        assert len(result.verification_pairs) == 2

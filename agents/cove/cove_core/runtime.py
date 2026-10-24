"""Runtime wiring for CoVe server."""

from __future__ import annotations

from dataclasses import dataclass

from cove_core.agents.factory import agents_use_fixture, build_cove_agents
from cove_core.config import CoVeConfig
from cove_core.orchestrator import CoVeOrchestrator


@dataclass
class CoveRuntime:
    """Handles for config and orchestrator."""

    config: CoVeConfig
    orchestrator: CoVeOrchestrator
    agents_mode: str

    def close(self) -> None:
        """Release external resources (none in scaffold)."""


def build_runtime(config: CoVeConfig) -> CoveRuntime:
    """Construct runtime from config."""
    drafter, planner, executor, reviser = build_cove_agents()
    orchestrator = CoVeOrchestrator(
        config,
        drafter=drafter,
        planner=planner,
        executor=executor,
        reviser=reviser,
    )
    mode = "fixture" if agents_use_fixture() else "llm"
    return CoveRuntime(config=config, orchestrator=orchestrator, agents_mode=mode)

"""CoVe orchestrator: four steps with isolated verification execution."""

from __future__ import annotations

from cove_core.config import CoVeConfig
from cove_core.models import CoVeRunResult
from cove_core.protocols import (
    BaselineDrafter,
    FinalReviser,
    VerificationExecutor,
    VerificationPlanner,
)


class CoVeOrchestrator:
    """Runs baseline → plan → execute (isolated) → revise."""

    def __init__(
        self,
        config: CoVeConfig,
        *,
        drafter: BaselineDrafter,
        planner: VerificationPlanner,
        executor: VerificationExecutor,
        reviser: FinalReviser,
    ) -> None:
        """Wire the four CoVe roles."""
        self._config = config
        self._drafter = drafter
        self._planner = planner
        self._executor = executor
        self._reviser = reviser

    async def run(self, query: str, *, query_id: str | None = None) -> CoVeRunResult:
        """Execute the full Chain-of-Verification pipeline."""
        baseline = self._drafter.draft(query, query_id=query_id)
        questions = self._planner.plan(query, baseline, query_id=query_id)
        pairs = self._executor.execute(query, questions, query_id=query_id)
        final = self._reviser.revise(query, baseline, pairs, query_id=query_id)
        return CoVeRunResult(
            query=query,
            query_id=query_id,
            baseline_response=baseline,
            verification_questions=questions,
            verification_pairs=pairs,
            final_response=final,
            execution_mode=self._config.verification_execution_mode,
        )

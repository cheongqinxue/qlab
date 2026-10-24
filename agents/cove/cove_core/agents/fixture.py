"""Fixture CoVe agents backed by ``eval/`` JSON."""

from __future__ import annotations

from cove_core.eval_loader import load_toy_pipeline
from cove_core.models import VerificationPair


class FixtureBaselineDrafter:
    """Return committed baseline for a toy ``query_id``."""

    def draft(self, query: str, *, query_id: str | None = None) -> str:
        """Load baseline from eval fixture."""
        _ = query
        if query_id is None:
            msg = "Fixture drafter requires query_id"
            raise ValueError(msg)
        return str(load_toy_pipeline(query_id)["baseline_response"])


class FixtureVerificationPlanner:
    """Return committed verification questions."""

    def plan(self, query: str, baseline: str, *, query_id: str | None = None) -> list[str]:
        """Load questions from eval fixture."""
        _ = query
        _ = baseline
        if query_id is None:
            msg = "Fixture planner requires query_id"
            raise ValueError(msg)
        return list(load_toy_pipeline(query_id)["verification_questions"])


class FixtureVerificationExecutor:
    """Return committed isolated Q/A pairs (ignores live questions text in scaffold)."""

    def execute(
        self,
        query: str,
        questions: list[str],
        *,
        query_id: str | None = None,
    ) -> list[VerificationPair]:
        """Load pairs from eval fixture."""
        _ = query
        _ = questions
        if query_id is None:
            msg = "Fixture executor requires query_id"
            raise ValueError(msg)
        rows = load_toy_pipeline(query_id)["verification_pairs"]
        return [VerificationPair.model_validate(row) for row in rows]


class FixtureFinalReviser:
    """Return committed final verified response."""

    def revise(
        self,
        query: str,
        baseline: str,
        pairs: list[VerificationPair],
        *,
        query_id: str | None = None,
    ) -> str:
        """Load final response from eval fixture."""
        _ = query
        _ = baseline
        _ = pairs
        if query_id is None:
            msg = "Fixture reviser requires query_id"
            raise ValueError(msg)
        return str(load_toy_pipeline(query_id)["final_response"])

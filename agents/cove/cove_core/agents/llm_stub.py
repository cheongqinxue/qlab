"""Placeholder LLM agents for CoVe (Phase 2)."""

from __future__ import annotations

from cove_core.models import VerificationPair


class LlmCoVeNotImplementedError(NotImplementedError):
    """Raised when ``COVE_AGENTS=llm`` before Phase 2."""


class LlmBaselineDrafter:
    """LLM baseline drafter (stub)."""

    def draft(self, query: str, *, query_id: str | None = None) -> str:
        """Raise until LLM prompts are wired."""
        _ = query
        _ = query_id
        raise LlmCoVeNotImplementedError("COVE LLM baseline drafter not implemented")


class LlmVerificationPlanner:
    """LLM verification planner (stub)."""

    def plan(self, query: str, baseline: str, *, query_id: str | None = None) -> list[str]:
        """Raise until LLM prompts are wired."""
        _ = query
        _ = baseline
        _ = query_id
        raise LlmCoVeNotImplementedError("COVE LLM verification planner not implemented")


class LlmVerificationExecutor:
    """LLM factored executor (stub)."""

    def execute(
        self,
        query: str,
        questions: list[str],
        *,
        query_id: str | None = None,
    ) -> list[VerificationPair]:
        """Raise until LLM prompts are wired."""
        _ = query
        _ = questions
        _ = query_id
        raise LlmCoVeNotImplementedError("COVE LLM verification executor not implemented")


class LlmFinalReviser:
    """LLM final reviser (stub)."""

    def revise(
        self,
        query: str,
        baseline: str,
        pairs: list[VerificationPair],
        *,
        query_id: str | None = None,
    ) -> str:
        """Raise until LLM prompts are wired."""
        _ = query
        _ = baseline
        _ = pairs
        _ = query_id
        raise LlmCoVeNotImplementedError("COVE LLM final reviser not implemented")

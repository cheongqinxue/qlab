"""CoVe role protocols (four LLM steps)."""

from __future__ import annotations

from typing import Protocol

from cove_core.models import VerificationPair


class BaselineDrafter(Protocol):
    """Step 1: initial response to the user query."""

    def draft(self, query: str, *, query_id: str | None = None) -> str:
        """Return baseline model response."""


class VerificationPlanner(Protocol):
    """Step 2: verification questions for claims in the baseline."""

    def plan(self, query: str, baseline: str, *, query_id: str | None = None) -> list[str]:
        """Return short factual questions (order preserved)."""


class VerificationExecutor(Protocol):
    """Step 3: answer verification questions without the baseline in context."""

    def execute(
        self,
        query: str,
        questions: list[str],
        *,
        query_id: str | None = None,
    ) -> list[VerificationPair]:
        """Return isolated Q/A pairs (one per question for factored mode)."""


class FinalReviser(Protocol):
    """Step 4: revised response using verification evidence."""

    def revise(
        self,
        query: str,
        baseline: str,
        pairs: list[VerificationPair],
        *,
        query_id: str | None = None,
    ) -> str:
        """Return final verified response."""

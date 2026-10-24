"""CoVe LLM agents with injected runners (no API key)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from cove_core.agents.llm import (
    LlmBaselineDrafter,
    LlmFinalReviser,
    LlmVerificationExecutor,
    LlmVerificationPlanner,
)
from cove_core.agents.llm_models import VerificationPlan
from cove_core.config import CoVeConfig
from cove_core.orchestrator import CoVeOrchestrator


class TestLlmAgents:
    """Unit tests for LLM step agents."""

    def test_executor_does_not_receive_baseline_in_prompt(self) -> None:
        captured: list[str] = []

        class RecordingModel:
            def invoke(self, messages, config=None):  # noqa: ANN001, ARG002
                captured.append(messages[-1].content)
                return SimpleNamespace(content="1848")

        executor = LlmVerificationExecutor(chat_model=RecordingModel())
        pairs = executor.execute(
            "When did the war end?",
            ["When did the Mexican–American War end?"],
        )
        assert len(pairs) == 1
        assert "1848" in pairs[0].answer
        assert all("Draft" not in text for text in captured)
        assert all("1849" not in text for text in captured)

    def test_orchestrator_with_mock_llm_chain(self) -> None:
        class BaselineModel:
            def invoke(self, messages, config=None):  # noqa: ANN001, ARG002
                return SimpleNamespace(content="Draft says 1846 to 1849.")

        class ExecuteModel:
            def invoke(self, messages, config=None):  # noqa: ANN001, ARG002
                return SimpleNamespace(content="The war ended in 1848.")

        class ReviseModel:
            def invoke(self, messages, config=None):  # noqa: ANN001, ARG002
                return SimpleNamespace(content="Corrected: 1846 to 1848.")

        class PlanRunner:
            def invoke(self, messages, config=None):  # noqa: ANN001, ARG002
                return VerificationPlan(
                    questions=["When did the Mexican–American War end?"],
                )

        orchestrator = CoVeOrchestrator(
            CoVeConfig(),
            drafter=LlmBaselineDrafter(chat_model=BaselineModel()),
            planner=LlmVerificationPlanner(runner=PlanRunner()),
            executor=LlmVerificationExecutor(chat_model=ExecuteModel()),
            reviser=LlmFinalReviser(chat_model=ReviseModel()),
        )
        result = asyncio.run(
            orchestrator.run("Summarize the Mexican–American War dates."),
        )
        assert "1849" in result.baseline_response
        assert "1848" in result.final_response
        assert len(result.verification_pairs) == 1

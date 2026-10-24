"""LLM implementations for the four CoVe steps (factored execution)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from cove_core.agents.llm_client import create_chat_model
from agents_utils.observability.langfuse_tracing import langfuse_task_session

from cove_core.agents.llm_invoke import invoke_text, langchain_invoke_config
from cove_core.agents.llm_models import VerificationPlan
from cove_core.agents.prompts import (
    BASELINE_HUMAN,
    BASELINE_SYSTEM,
    EXECUTE_HUMAN,
    EXECUTE_SYSTEM,
    PLAN_HUMAN,
    PLAN_SYSTEM,
    REVISE_HUMAN,
    REVISE_SYSTEM,
)
from cove_core.models import VerificationPair


def _format_pairs_text(pairs: list[VerificationPair]) -> str:
    lines = [f"Q: {pair.question}\nA: {pair.answer}" for pair in pairs]
    return "\n\n".join(lines)


class LlmBaselineDrafter:
    """Step 1: baseline LLM response."""

    def __init__(self, model: str | None = None, *, chat_model: object | None = None) -> None:
        """Bind chat model (injectable for tests)."""
        self._model = chat_model if chat_model is not None else create_chat_model(model)

    def draft(self, query: str, *, query_id: str | None = None) -> str:
        """Return baseline model response."""
        _ = query_id
        return invoke_text(
            self._model,
            system=BASELINE_SYSTEM,
            human=BASELINE_HUMAN.format(query=query),
            session_id="cove-baseline",
        )


class LlmVerificationPlanner:
    """Step 2: verification questions from query + baseline."""

    def __init__(self, model: str | None = None, *, runner: object | None = None) -> None:
        """Bind structured planner runner (injectable for tests)."""
        if runner is None:
            runner = create_chat_model(model).with_structured_output(VerificationPlan)
        self._runner = runner

    def plan(self, query: str, baseline: str, *, query_id: str | None = None) -> list[str]:
        """Return short factual questions."""
        _ = query_id
        human = PLAN_HUMAN.format(query=query, baseline=baseline)
        with langfuse_task_session(task_id="cove-plan", task_preview=human):
            plan: VerificationPlan = self._runner.invoke(
                [
                    SystemMessage(content=PLAN_SYSTEM),
                    HumanMessage(content=human),
                ],
                config=langchain_invoke_config(),
            )
        return [q.strip() for q in plan.questions if q.strip()]


class LlmVerificationExecutor:
    """Step 3: factored isolated answers (no baseline in prompt)."""

    def __init__(self, model: str | None = None, *, chat_model: object | None = None) -> None:
        """Bind chat model (injectable for tests)."""
        self._model = chat_model if chat_model is not None else create_chat_model(model)

    def execute(
        self,
        query: str,
        questions: list[str],
        *,
        query_id: str | None = None,
    ) -> list[VerificationPair]:
        """Return one isolated answer per verification question."""
        _ = query
        _ = query_id
        pairs: list[VerificationPair] = []
        for index, question in enumerate(questions):
            answer = invoke_text(
                self._model,
                system=EXECUTE_SYSTEM,
                human=EXECUTE_HUMAN.format(question=question),
                session_id=f"cove-execute-{index}",
            )
            pairs.append(VerificationPair(question=question, answer=answer))
        return pairs


class LlmFinalReviser:
    """Step 4: revised response using verification evidence."""

    def __init__(self, model: str | None = None, *, chat_model: object | None = None) -> None:
        """Bind chat model (injectable for tests)."""
        self._model = chat_model if chat_model is not None else create_chat_model(model)

    def revise(
        self,
        query: str,
        baseline: str,
        pairs: list[VerificationPair],
        *,
        query_id: str | None = None,
    ) -> str:
        """Return final verified response."""
        _ = query_id
        pairs_text = _format_pairs_text(pairs)
        return invoke_text(
            self._model,
            system=REVISE_SYSTEM,
            human=REVISE_HUMAN.format(query=query, baseline=baseline, pairs_text=pairs_text),
            session_id="cove-revise",
        )

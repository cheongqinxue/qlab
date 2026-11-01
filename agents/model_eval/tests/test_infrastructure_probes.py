"""Infrastructure probes from stored messages."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from model_eval_core.dimensions import EvalDimension
from model_eval_core.infrastructure_probes import (
    FAILURE_MODE_REPETITION_LOOP,
    FAILURE_MODE_THINKING_BUDGET,
    append_infrastructure_probes,
    detect_repetition_loop,
    max_completion_tokens,
)
from model_eval_core.probes import ProbeResult, probe_case


class TestInfrastructureProbes:
    """Thinking budget and repetition heuristics."""

    def test_max_completion_tokens_from_serialized_messages(self) -> None:
        messages = [
            {"type": "human", "content": "hi"},
            {"type": "ai", "content": "x", "usage": {"completion_tokens": 1200}},
            {"type": "ai", "content": "y", "usage": {"completion_tokens": 32000}},
        ]
        assert max_completion_tokens(messages) == 32000

    def test_thinking_budget_exhausted_on_empty_visible(self) -> None:
        raw = "<think>\n" + ("analyze " * 50) + "\n</think>"
        messages = [{"type": "ai", "content": raw, "usage": {"completion_tokens": 32768}}]
        probe = ProbeResult()
        append_infrastructure_probes(
            probe,
            case_id="truthfulqa-fortune-cookie-001",
            final_answer_raw=raw,
            messages=messages,
        )
        assert FAILURE_MODE_THINKING_BUDGET in probe.failure_modes
        assert probe.signals["infrastructure"]["max_completion_tokens"] == 32768

    def test_repetition_skips_ifeval_repeat_prompt(self) -> None:
        repeated = ("Repeat the request.\n" * 8).strip()
        assert not detect_repetition_loop(
            repeated,
            case_id="ifeval-repeat-title-003",
            metadata={"instruction_id_list": ["combination:repeat_prompt"]},
        )

    def test_repetition_detects_loops(self) -> None:
        line = "The same sentence appears again and again here."
        text = "\n".join([line] * 5)
        assert detect_repetition_loop(text, case_id="math-prime-roots-020")

    def test_probe_case_adds_infrastructure(self) -> None:
        loop_line = "Mode collapse sample text for testing repetition detection."
        body = "\n".join([loop_line] * 5)
        messages = [HumanMessage(content="q"), AIMessage(content=body)]
        probe = probe_case(
            dimension=EvalDimension.MATH,
            prompt="q",
            messages=messages,
            metadata={"solution": "\\boxed{99}"},
            case_id="math-test-001",
        )
        assert FAILURE_MODE_REPETITION_LOOP in probe.failure_modes

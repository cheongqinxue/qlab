"""Failure-mode probes."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from model_eval_core.dimensions import EvalDimension
from model_eval_core.probes import probe_case


class TestProbeCase:
    """Heuristic failure tagging."""

    def test_tool_use_flags_missing_expected_tool(self) -> None:
        messages = [
            HumanMessage(content="use weather tool"),
            AIMessage(content="It is sunny."),
        ]
        probe = probe_case(
            dimension=EvalDimension.TOOL_USE,
            prompt="Use the get_weather tool for Paris.",
            messages=messages,
            metadata={"expected_tools": ["get_weather"]},
        )
        assert "missing_expected_tool" in probe.failure_modes

    def test_coding_requires_entry_point_even_after_tool_calls(self) -> None:
        messages = [
            HumanMessage(content="implement skjkasdkd"),
            AIMessage(content="Sorry, need more steps to process this request."),
        ]
        probe = probe_case(
            dimension=EvalDimension.CODING,
            prompt="def skjkasdkd(lst): ...",
            messages=messages,
            metadata={"entry_point": "skjkasdkd"},
        )
        assert "missing_entry_point_in_answer" in probe.failure_modes
        assert "agent_step_limit_stub" in probe.failure_modes

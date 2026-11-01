"""Execution mode selection."""

from __future__ import annotations

from model_eval_core.dimensions import EvalDimension
from model_eval_core.execution_mode import uses_react_tools


class TestExecutionMode:
    """ReAct is bound to tool_use only."""

    def test_only_tool_use_uses_react(self) -> None:
        assert uses_react_tools(EvalDimension.TOOL_USE) is True
        for dim in EvalDimension:
            if dim != EvalDimension.TOOL_USE:
                assert uses_react_tools(dim) is False

"""How a curated case is executed (ReAct+tools vs plain chat)."""

from __future__ import annotations

from model_eval_core.dimensions import EvalDimension


def uses_react_tools(dimension: EvalDimension) -> bool:
    """Return True only for dimensions that require the tool-bound ReAct agent."""
    return dimension == EvalDimension.TOOL_USE

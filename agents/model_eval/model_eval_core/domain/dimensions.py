"""Evaluation dimensions, cheatsheet labels, and harness mode helpers."""

from __future__ import annotations

from enum import StrEnum


class EvalDimension(StrEnum):
    """Agent eval axes aligned with curated benchmark slices."""

    TOOL_USE = "tool_use"
    INSTRUCTION_FOLLOWING = "instruction_following"
    HALLUCINATION = "hallucination"
    SYCOPHANCY = "sycophancy"
    REASONING = "reasoning"
    MATH = "math"
    CODING = "coding"


DIMENSION_LABELS: dict[EvalDimension, str] = {
    EvalDimension.TOOL_USE: "Tool use accuracy",
    EvalDimension.INSTRUCTION_FOLLOWING: "Instruction following",
    EvalDimension.HALLUCINATION: "Hallucination resistance",
    EvalDimension.SYCOPHANCY: "Sycophancy resistance",
    EvalDimension.REASONING: "Reasoning / problem solving",
    EvalDimension.MATH: "Math",
    EvalDimension.CODING: "Coding",
}

CHEATSHEET_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("tool_use", "Tool use"),
    ("instruction_following", "Instruction following"),
    ("hallucination", "Hallucination"),
    ("sycophancy", "Sycophancy"),
    ("reasoning", "Reasoning"),
    ("math", "Math"),
    ("coding", "Coding"),
)

QUALITATIVE_DIMENSIONS: frozenset[EvalDimension] = frozenset(
    {
        EvalDimension.INSTRUCTION_FOLLOWING,
        EvalDimension.HALLUCINATION,
        EvalDimension.SYCOPHANCY,
    }
)

INFRASTRUCTURE_FAILURE_MODES: tuple[tuple[str, str], ...] = (
    ("ollama_tool_binding_unsupported", "No tool API"),
    ("missing_expected_tool", "Missing tool"),
    ("tool_required_but_not_called", "Tool not called"),
    ("malformed_tool_call_syntax", "Bad tool syntax"),
    ("agent_step_limit_stub", "Step limit"),
    ("thinking_budget_exhausted", "Think budget"),
    ("repetition_loop", "Repetition"),
    ("empty_final_answer", "Empty answer"),
)

INFRASTRUCTURE_MODE_IDS: frozenset[str] = frozenset(
    mode for mode, _ in INFRASTRUCTURE_FAILURE_MODES
)


def is_qualitative(dimension: EvalDimension) -> bool:
    """Return True when pass/fail comes from human review, not deterministic probes."""
    return dimension in QUALITATIVE_DIMENSIONS


def uses_react_tools(dimension: EvalDimension) -> bool:
    """Return True only for dimensions that require the tool-bound ReAct agent."""
    return dimension == EvalDimension.TOOL_USE

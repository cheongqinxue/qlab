"""Evaluation dimensions and benchmark metadata."""

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

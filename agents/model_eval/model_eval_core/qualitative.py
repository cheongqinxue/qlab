"""Dimensions scored by human review, not deterministic probes."""

from __future__ import annotations

from model_eval_core.dimensions import EvalDimension

# Instruction-following (IFEval-style), TruthfulQA-style hallucination, sycophancy probes.
QUALITATIVE_DIMENSIONS: frozenset[EvalDimension] = frozenset(
    {
        EvalDimension.INSTRUCTION_FOLLOWING,
        EvalDimension.HALLUCINATION,
        EvalDimension.SYCOPHANCY,
    }
)


def is_qualitative(dimension: EvalDimension) -> bool:
    """Return True when pass/fail must come from human review, not regex probes."""
    return dimension in QUALITATIVE_DIMENSIONS

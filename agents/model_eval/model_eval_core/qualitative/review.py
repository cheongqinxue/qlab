"""Human-reviewed dimensions and committed pass/fail lookup."""

from __future__ import annotations

import json
from pathlib import Path

from model_eval_core.data.paths import HUMAN_REVIEW_PATH
from model_eval_core.domain.dimensions import QUALITATIVE_DIMENSIONS, EvalDimension, is_qualitative

__all__ = [
    "QUALITATIVE_DIMENSIONS",
    "EvalDimension",
    "is_qualitative",
    "load_human_review",
]


def load_human_review(path: Path | None = None) -> dict[tuple[str, str], bool]:
    """Map (model_dir, case_id) to human pass boolean.

    Args:
        path: JSONL path; defaults to ``eval/human_qualitative_review.jsonl``.

    Returns:
        Lookup table; missing keys mean not yet reviewed.
    """
    review_path = path or HUMAN_REVIEW_PATH
    if not review_path.exists():
        return {}
    table: dict[tuple[str, str], bool] = {}
    with review_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            table[(str(row["model_dir"]), str(row["case_id"]))] = bool(row["passed"])
    return table

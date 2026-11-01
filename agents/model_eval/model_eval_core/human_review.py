"""Load committed human pass/fail for qualitative eval dimensions."""

from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "eval" / "human_qualitative_review.jsonl"


def load_human_review(path: Path | None = None) -> dict[tuple[str, str], bool]:
    """Map (model_dir, case_id) to human pass boolean.

    Args:
        path: JSONL path; defaults to eval/human_qualitative_review.jsonl.

    Returns:
        Lookup table; missing keys mean not yet reviewed.
    """
    review_path = path or _DEFAULT_PATH
    if not review_path.exists():
        return {}
    table: dict[tuple[str, str], bool] = {}
    with review_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (str(row["model_dir"]), str(row["case_id"]))
            table[key] = bool(row["passed"])
    return table

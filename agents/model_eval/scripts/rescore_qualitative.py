#!/usr/bin/env python3
"""Apply human_qualitative_review.jsonl to responses.jsonl and rebuild summary.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
RESULTS = _REPO / "agents" / "model_eval" / "eval" / "results"
REVIEW_PATH = _REPO / "agents" / "model_eval" / "eval" / "human_qualitative_review.jsonl"

sys.path.insert(0, str(_REPO / "agents" / "model_eval"))
from model_eval_core.human_review import load_human_review  # noqa: E402
from model_eval_core.qualitative import QUALITATIVE_DIMENSIONS  # noqa: E402


def rescore_model_dir(model_dir: Path, review: dict[tuple[str, str], bool]) -> None:
    """Update passed flags and summary for one model directory."""
    responses_path = model_dir / "responses.jsonl"
    summary_path = model_dir / "summary.json"
    if not responses_path.exists() or not summary_path.exists():
        return
    model_key = model_dir.name
    lines: list[str] = []
    by_dim: dict[str, dict[str, int]] = {}
    total_pass = 0
    total_cases = 0

    for line in responses_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        dim = row["dimension"]
        case_id = row["case_id"]
        total_cases += 1
        if dim in {d.value for d in QUALITATIVE_DIMENSIONS}:
            key = (model_key, case_id)
            fm = row.get("failure_modes") or []
            final = str(row.get("final_answer") or "")
            infra_failed = "runtime_error" in fm or final.startswith("[ERROR]")
            if infra_failed:
                row["passed"] = False
                row["human_reviewed"] = False
            elif key in review:
                row["passed"] = review[key]
                row["human_reviewed"] = True
            else:
                row["human_reviewed"] = False
        if row.get("passed"):
            total_pass += 1
        block = by_dim.setdefault(dim, {"passed": 0, "total": 0})
        block["total"] += 1
        if row.get("passed"):
            block["passed"] += 1
        lines.append(json.dumps(row, ensure_ascii=False))

    responses_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["total_passed"] = total_pass
    summary["total_cases"] = total_cases
    summary["by_dimension"] = by_dim
    summary["human_qualitative_review"] = True
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Rescore all model result folders."""
    review = load_human_review(REVIEW_PATH)
    if not review:
        raise SystemExit(f"No review rows in {REVIEW_PATH}")
    for model_dir in sorted(RESULTS.iterdir()):
        if model_dir.is_dir():
            rescore_model_dir(model_dir, review)
    print(f"Rescored qualitative dimensions using {len(review)} human review rows.")


if __name__ == "__main__":
    main()

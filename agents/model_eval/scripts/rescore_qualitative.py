#!/usr/bin/env python3
"""Apply human_qualitative_review.jsonl to responses.jsonl and rebuild summary.json."""

from __future__ import annotations

import json
from pathlib import Path

from model_eval_core.data.paths import HUMAN_REVIEW_PATH, RESULTS_DIR, install_import_path

install_import_path()

from model_eval_core.domain import QUALITATIVE_DIMENSIONS  # noqa: E402
from model_eval_core.qualitative import load_human_review  # noqa: E402


def rescore_model_dir(model_dir: Path, review: dict[tuple[str, str], bool]) -> None:
    """Update passed flags and summary for one model directory."""
    responses_path = model_dir / "responses.jsonl"
    summary_path = model_dir / "summary.json"
    if not responses_path.exists() or not summary_path.exists():
        return
    model_key = model_dir.name
    lines_out: list[str] = []
    by_dim: dict[str, dict[str, int]] = {}
    total_pass = total_cases = 0

    for line in responses_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total_cases += 1
        dim = row.get("dimension", "")
        case_id = str(row.get("case_id", ""))
        if dim in {d.value for d in QUALITATIVE_DIMENSIONS}:
            key = (model_key, case_id)
            if key in review:
                row["passed"] = review[key]
                row["human_reviewed"] = True
            else:
                row["human_reviewed"] = False
        block = by_dim.setdefault(dim, {"passed": 0, "total": 0})
        block["total"] += 1
        if row.get("passed"):
            block["passed"] += 1
            total_pass += 1
        lines_out.append(json.dumps(row, ensure_ascii=False))

    responses_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["total_passed"] = total_pass
    summary["total_cases"] = total_cases
    summary["by_dimension"] = by_dim
    summary["human_qualitative_review"] = True
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Rescore all model result folders."""
    review = load_human_review(HUMAN_REVIEW_PATH)
    if not review:
        raise SystemExit(f"No review rows in {HUMAN_REVIEW_PATH}")
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if model_dir.is_dir():
            rescore_model_dir(model_dir, review)
    print(f"Rescored qualitative dimensions using {len(review)} human review rows.")


if __name__ == "__main__":
    main()

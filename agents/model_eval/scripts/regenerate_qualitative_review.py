#!/usr/bin/env python3
"""Rebuild human_qualitative_review.jsonl (verbatim auto-grade + assistant overrides)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "agents"))
sys.path.insert(0, str(_REPO / "agents" / "model_eval"))

from model_eval_core.prompts import read_curated_jsonl
from model_eval_core.qualitative_rubric import (
    assistant_judge_hallucination,
    hallucination_auto_grade,
    score_instruction_following,
    score_sycophancy,
)

RESULTS = _REPO / "agents" / "model_eval" / "eval" / "results"
PROMPTS = _REPO / "agents" / "model_eval" / "eval" / "prompts"
OUT = _REPO / "agents" / "model_eval" / "eval" / "human_qualitative_review.jsonl"
ASSISTANT_OVERRIDES = (
    _REPO / "agents" / "model_eval" / "eval" / "hallucination_assistant_review.json"
)
QUAL = frozenset({"instruction_following", "hallucination", "sycophancy"})


def _load_assistant_overrides() -> dict[str, bool]:
    """Map ``model_dir/case_id`` to assistant pass for non-verbatim hallucination."""
    if not ASSISTANT_OVERRIDES.exists():
        return {}
    data = json.loads(ASSISTANT_OVERRIDES.read_text(encoding="utf-8"))
    return {str(k): bool(v) for k, v in data.items()}


def _infra_fail(answer: str, failure_modes: list) -> bool:
    return (
        not answer.strip()
        or "empty_final_answer" in failure_modes
        or answer.startswith("[ERROR]")
        or answer.startswith("[N/A]")
    )


def main() -> None:
    """Write review rows; print hallucination rows still needing assistant review."""
    meta_by_id: dict[str, tuple[str, dict]] = {}
    for dim in QUAL:
        for row in read_curated_jsonl(PROMPTS / f"{dim}.jsonl"):
            meta_by_id[row.id] = (dim, dict(row.metadata))

    assistant = _load_assistant_overrides()
    records: list[dict] = []
    for model_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        responses_path = model_dir / "responses.jsonl"
        if not responses_path.exists():
            continue
        for line in responses_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["dimension"] not in QUAL:
                continue
            case_id = str(row["case_id"])
            dim, meta = meta_by_id[case_id]
            answer = str(row.get("final_answer") or "")
            failure_modes = row.get("failure_modes") or []
            override_key = f"{model_dir.name}/{case_id}"

            if _infra_fail(answer, failure_modes):
                passed, note = False, "empty or infra failure"
            elif dim == "hallucination":
                auto = hallucination_auto_grade(answer, meta)
                if auto is not None:
                    passed, note = auto, "auto verbatim hallucination"
                elif override_key in assistant:
                    passed = assistant[override_key]
                    note = "assistant paraphrase override"
                else:
                    passed = assistant_judge_hallucination(case_id, answer, meta)
                    note = "assistant paraphrase review"
            elif dim == "instruction_following":
                passed = score_instruction_following(answer, meta)
                note = "ifeval structure rubric"
            elif dim == "sycophancy":
                passed = score_sycophancy(answer, meta)
                note = "sycophancy rubric"
            records.append(
                {
                    "model_dir": model_dir.name,
                    "case_id": case_id,
                    "dimension": dim,
                    "passed": passed,
                    "reviewer": "cursor_agent",
                    "note": note,
                }
            )

    OUT.write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} rows to {OUT}")


if __name__ == "__main__":
    main()

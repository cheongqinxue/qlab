"""Standalone coding-dimension evaluation (HumanEval exec), decoupled from the LLM harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from model_eval_core.dimensions import EvalDimension
from model_eval_core.probes import ProbeResult, probe_case
from model_eval_core.prompts import read_curated_jsonl
from model_eval_core.scoring import case_passed

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "eval" / "prompts"
_CODING_JSONL = _PROMPTS_DIR / "coding.jsonl"


def load_coding_metadata_by_case_id(
    prompts_path: Path = _CODING_JSONL,
) -> dict[str, dict[str, Any]]:
    """Load HumanEval metadata keyed by curated case id.

    Args:
        prompts_path: Path to ``coding.jsonl``.

    Returns:
        Mapping of case id to metadata dict (entry_point, test, etc.).
    """
    out: dict[str, dict[str, Any]] = {}
    for row in read_curated_jsonl(prompts_path):
        out[row.id] = dict(row.metadata)
        out[row.id]["_prompt"] = row.prompt
    return out


def probe_coding_case(
    *,
    prompt: str,
    answer: str,
    metadata: dict[str, Any],
) -> ProbeResult:
    """Build a full :class:`ProbeResult` for the coding dimension from stored text.

    Args:
        prompt: User prompt sent to the model.
        answer: Final model answer text.
        metadata: HumanEval metadata from ``coding.jsonl``.

    Returns:
        Probe result including ``humaneval_exec`` signals when tests run.
    """
    messages = [
        HumanMessage(content=prompt),
        AIMessage(content=answer),
    ]
    return probe_case(
        dimension=EvalDimension.CODING,
        prompt=prompt,
        messages=messages,
        metadata=metadata,
    )


def coding_case_passed(
    probe: ProbeResult,
    *,
    model_dir: str | None = None,
    case_id: str | None = None,
) -> bool:
    """Return pass/fail for a coding probe (same rules as live harness)."""
    return case_passed(
        probe,
        dimension=EvalDimension.CODING,
        model_dir=model_dir,
        case_id=case_id,
    )


def rescore_coding_row(
    row: dict[str, Any],
    metadata: dict[str, Any],
    *,
    model_dir: str,
) -> dict[str, Any]:
    """Re-score one ``responses.jsonl`` coding row using saved ``final_answer``.

    Args:
        row: Parsed JSONL record from a prior eval run.
        metadata: HumanEval metadata for ``row['case_id']``.
        model_dir: Results folder name (for pass/fail hooks).

    Returns:
        Updated row dict (mutates and returns the same mapping).
    """
    prompt = str(row.get("prompt") or metadata.get("_prompt") or "")
    answer = str(row.get("final_answer") or "")
    probe = probe_coding_case(prompt=prompt, answer=answer, metadata=metadata)
    row["failure_modes"] = probe.failure_modes
    prior = row.get("signals") if isinstance(row.get("signals"), dict) else {}
    row["signals"] = {**prior, **probe.signals}
    row["passed"] = coding_case_passed(
        probe,
        model_dir=model_dir,
        case_id=str(row.get("case_id", "")),
    )
    row["coding_rescored"] = True
    return row


def rescore_model_coding_dir(
    model_dir: Path,
    metadata_by_id: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    """Re-score all coding rows in one model results folder.

    Args:
        model_dir: Path to ``eval/results/<name>/``.
        metadata_by_id: From :func:`load_coding_metadata_by_case_id`.

    Returns:
        Tuple of (coding cases updated, coding cases passed after rescore).
    """
    responses_path = model_dir / "responses.jsonl"
    summary_path = model_dir / "summary.json"
    if not responses_path.exists():
        return 0, 0

    lines_out: list[str] = []
    updated = 0
    coding_pass = 0
    coding_total = 0
    total_pass = 0
    total_cases = 0
    by_dim: dict[str, dict[str, int]] = {}

    for line in responses_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total_cases += 1
        if row.get("dimension") == EvalDimension.CODING.value:
            case_id = str(row.get("case_id", ""))
            meta = metadata_by_id.get(case_id)
            if meta is not None:
                rescore_coding_row(row, meta, model_dir=model_dir.name)
                updated += 1
            coding_total += 1
            if row.get("passed"):
                coding_pass += 1
        dim = str(row.get("dimension", ""))
        block = by_dim.setdefault(dim, {"passed": 0, "total": 0})
        block["total"] += 1
        if row.get("passed"):
            block["passed"] += 1
            total_pass += 1
        lines_out.append(json.dumps(row, ensure_ascii=False))

    responses_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["total_passed"] = total_pass
        summary["total_cases"] = total_cases
        summary["by_dimension"] = by_dim
        summary["coding_humaneval_rescored"] = True
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return updated, coding_pass

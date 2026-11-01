"""Extract model code and run HumanEval ``check()`` in a subprocess after static validation."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_eval_core.coding.safety import validate_program_safety
from model_eval_core.data.io import read_json, read_jsonl, write_json, write_jsonl
from model_eval_core.domain.dimensions import EvalDimension

DEFAULT_TIMEOUT_S = float(os.environ.get("MODEL_EVAL_CODING_EXEC_TIMEOUT", "5"))
_AGENTS_ROOT = Path(__file__).resolve().parents[3]
_MODEL_EVAL_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HumanevalExecResult:
    """Result of executing HumanEval tests against extracted model code."""

    passed: bool
    blocked: bool
    safety_reason: str | None = None
    error: str | None = None
    extracted: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialize for probe signals."""
        return {
            "passed": self.passed,
            "blocked": self.blocked,
            "safety_reason": self.safety_reason,
            "error": self.error,
            "extracted": self.extracted,
        }


_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_python_from_answer(answer: str, entry_point: str) -> str | None:
    """Return the best Python block containing ``def entry_point``."""
    blocks = [block.strip() for block in _FENCE_RE.findall(answer) if block.strip()]
    for block in reversed(blocks):
        if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", block, re.MULTILINE):
            return block
    for block in reversed(blocks):
        if f"def {entry_point}" in block:
            return block
    return blocks[-1] if blocks else None


def _strip_markdown_prose(answer: str) -> str:
    """Drop fenced blocks and common markdown wrappers from free-form answers."""
    text = _FENCE_RE.sub("", answer)
    return text.strip()


def assemble_humaneval_program(
    *,
    prompt: str,
    answer: str,
    entry_point: str,
    test_code: str,
) -> tuple[str | None, bool]:
    """Build full source: solution + HumanEval tests + ``check(entry_point)`` call.

    Returns:
        Tuple of (program source or None if nothing usable, whether code was extracted).
    """
    extracted = extract_python_from_answer(answer, entry_point)
    has_def = bool(
        extracted
        and re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", extracted, re.MULTILINE)
    )
    if has_def and extracted is not None:
        solution = extracted
        was_extracted = True
    else:
        tail = extracted or _strip_markdown_prose(answer)
        if not tail:
            return None, False
        solution = f"{prompt.rstrip()}\n{tail}"
        was_extracted = extracted is not None

    program = f"{solution.rstrip()}\n\n{test_code.strip()}\n\ncheck({entry_point})\n"
    return program, was_extracted


def run_humaneval_check(
    *,
    prompt: str,
    answer: str,
    entry_point: str,
    test_code: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> HumanevalExecResult:
    """Validate assembled code, then run HumanEval tests in an isolated subprocess.

    Static validation always runs in this process before any execution.
    """
    program, extracted = assemble_humaneval_program(
        prompt=prompt,
        answer=answer,
        entry_point=entry_point,
        test_code=test_code,
    )
    if program is None:
        return HumanevalExecResult(
            passed=False,
            blocked=False,
            error="no_extractable_code",
            extracted=False,
        )

    safety = validate_program_safety(program)
    if not safety.ok:
        return HumanevalExecResult(
            passed=False,
            blocked=True,
            safety_reason=safety.reason,
            extracted=extracted,
        )

    pythonpath = os.pathsep.join(
        [str(_AGENTS_ROOT), str(_MODEL_EVAL_ROOT), os.environ.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    with tempfile.TemporaryDirectory(prefix="model_eval_humaneval_") as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "model_eval_core.coding.humaneval_runner"],
                input=program.encode("utf-8"),
                capture_output=True,
                timeout=timeout_s,
                cwd=tmp,
                env={
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": tmp,
                    "LANG": "C.UTF-8",
                    "PYTHONPATH": pythonpath,
                },
                check=False,
            )
        except subprocess.TimeoutExpired:
            return HumanevalExecResult(
                passed=False,
                blocked=False,
                error="execution_timeout",
                extracted=extracted,
            )

    if proc.returncode == 0:
        return HumanevalExecResult(passed=True, blocked=False, extracted=extracted)

    err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()
    blocked = proc.returncode == 2
    return HumanevalExecResult(
        passed=False,
        blocked=blocked,
        safety_reason=err if blocked else None,
        error=err[:500] if err else f"exit_{proc.returncode}",
        extracted=extracted,
    )


def evaluate_coding_answer(
    *,
    prompt: str,
    answer: str,
    metadata: dict[str, Any],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> HumanevalExecResult | None:
    """Run static safety + HumanEval ``check()`` for one model answer.

    Args:
        prompt: Curated task prompt (function stub).
        answer: Model completion text (may include markdown fences).
        metadata: Case metadata with ``entry_point`` and ``test``.
        timeout_s: Subprocess timeout for test execution.

    Returns:
        Execution result, or None when tests cannot be run (missing metadata).
    """
    entry_point = metadata.get("entry_point")
    test_code = metadata.get("test")
    if not entry_point or not test_code or not answer.strip():
        return None
    return run_humaneval_check(
        prompt=prompt,
        answer=answer,
        entry_point=str(entry_point),
        test_code=str(test_code),
        timeout_s=timeout_s,
    )


def rescore_coding_row(
    row: dict[str, Any],
    metadata: dict[str, Any],
    *,
    model_dir: str,
) -> dict[str, Any]:
    """Re-score one ``responses.jsonl`` coding row using saved ``final_answer``."""
    from langchain_core.messages import AIMessage, HumanMessage

    from model_eval_core.domain.scoring import case_passed
    from model_eval_core.probes.case import probe_case

    prompt = str(row.get("prompt") or metadata.get("_prompt") or "")
    answer = str(row.get("final_answer") or "")
    messages = [HumanMessage(content=prompt), AIMessage(content=answer)]
    probe = probe_case(
        dimension=EvalDimension.CODING,
        prompt=prompt,
        messages=messages,
        metadata=metadata,
        case_id=str(row.get("case_id", "")),
    )
    row["failure_modes"] = probe.failure_modes
    prior = row.get("signals") if isinstance(row.get("signals"), dict) else {}
    row["signals"] = {**prior, **probe.signals}
    row["passed"] = case_passed(
        probe,
        dimension=EvalDimension.CODING,
        model_dir=model_dir,
        case_id=str(row.get("case_id", "")),
    )
    row["coding_rescored"] = True
    return row


def rescore_model_coding_dir(
    model_dir: Path,
    metadata_by_id: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    """Re-score all coding rows in one model results folder."""
    responses_path = model_dir / "responses.jsonl"
    summary_path = model_dir / "summary.json"
    if not responses_path.exists():
        return 0, 0

    updated = coding_pass = coding_total = total_pass = total_cases = 0
    by_dim: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []

    for row in read_jsonl(responses_path):
        total_cases += 1
        if row.get("dimension") == EvalDimension.CODING.value:
            meta = metadata_by_id.get(str(row.get("case_id", "")))
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
        rows.append(row)

    write_jsonl(responses_path, rows)

    if summary_path.exists():
        summary = read_json(summary_path)
        summary.update(
            {
                "total_passed": total_pass,
                "total_cases": total_cases,
                "by_dimension": by_dim,
                "coding_humaneval_rescored": True,
            }
        )
        write_json(summary_path, summary)

    return updated, coding_pass

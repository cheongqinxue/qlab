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

from model_eval_core.coding_safety import validate_program_safety

DEFAULT_TIMEOUT_S = float(os.environ.get("MODEL_EVAL_CODING_EXEC_TIMEOUT", "5"))
_AGENTS_ROOT = Path(__file__).resolve().parents[2]
_MODEL_EVAL_ROOT = Path(__file__).resolve().parents[1]


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
                [sys.executable, "-m", "model_eval_core.humaneval_runner"],
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

"""Tests for HumanEval extraction, safety gate, and subprocess execution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from model_eval_core.coding_eval import rescore_coding_row
from model_eval_core.coding_exec import (
    assemble_humaneval_program,
    extract_python_from_answer,
    run_humaneval_check,
)
from model_eval_core.coding_safety import validate_program_safety
from model_eval_core.dimensions import EvalDimension
from model_eval_core.probes import probe_case

_CODING_JSONL = Path(__file__).resolve().parents[1] / "eval" / "prompts" / "coding.jsonl"


def _coding_row(case_id: str) -> dict:
    for line in _CODING_JSONL.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["id"] == case_id:
            return row
    raise KeyError(case_id)


class TestExtractPythonFromAnswer:
    """Markdown fence extraction."""

    def test_prefers_block_with_entry_point(self) -> None:
        answer = "text\n```python\ndef below_zero(x):\n    return False\n```\n"
        assert "def below_zero" in (extract_python_from_answer(answer, "below_zero") or "")


class TestRunHumanevalCheck:
    """Functional HumanEval checks using committed metadata."""

    def test_canonical_below_zero_passes(self) -> None:
        row = _coding_row("humaneval-below-zero-003")
        meta = row["metadata"]
        solution = meta["canonical_solution"]
        answer = f"```python\nfrom typing import List\n\n{row['prompt']}\n{solution}\n```"
        result = run_humaneval_check(
            prompt=row["prompt"],
            answer=answer,
            entry_point=meta["entry_point"],
            test_code=meta["test"],
        )
        assert result.blocked is False
        assert result.passed is True

    def test_wrong_solution_fails_tests(self) -> None:
        row = _coding_row("humaneval-below-zero-003")
        meta = row["metadata"]
        answer = (
            "```python\nfrom typing import List\n\n" + row["prompt"] + "\n    return False\n```"
        )
        result = run_humaneval_check(
            prompt=row["prompt"],
            answer=answer,
            entry_point=meta["entry_point"],
            test_code=meta["test"],
        )
        assert result.passed is False
        assert result.blocked is False

    def test_malicious_code_blocked_without_subprocess(self) -> None:
        row = _coding_row("humaneval-below-zero-003")
        meta = row["metadata"]
        malicious = "```python\nimport os\n" + row["prompt"] + "\n    os.system('rm -rf /')\n```"
        with patch("model_eval_core.coding_exec.subprocess.run") as mock_run:
            result = run_humaneval_check(
                prompt=row["prompt"],
                answer=malicious,
                entry_point=meta["entry_point"],
                test_code=meta["test"],
            )
            mock_run.assert_not_called()
        assert result.blocked is True
        assert result.passed is False

    def test_assembled_program_validates_before_run(self) -> None:
        row = _coding_row("humaneval-filter-integers-022")
        meta = row["metadata"]
        program, _ = assemble_humaneval_program(
            prompt=row["prompt"],
            answer=f"```python\n{row['prompt']}\n{meta['canonical_solution']}\n```",
            entry_point=meta["entry_point"],
            test_code=meta["test"],
        )
        assert program is not None
        assert validate_program_safety(program).ok is True


class TestProbeCaseCoding:
    """Coding dimension uses HumanEval execution when tests are present."""

    def test_passes_when_tests_pass_without_name_in_prose(self) -> None:
        row = _coding_row("humaneval-below-zero-003")
        meta = row["metadata"]
        body = meta["canonical_solution"]
        answer = f"```python\nfrom typing import List\n\n{row['prompt']}\n{body}\n```"
        messages = [
            HumanMessage(content=row["prompt"]),
            AIMessage(content=answer),
        ]
        probe = probe_case(
            dimension=EvalDimension.CODING,
            prompt=row["prompt"],
            messages=messages,
            metadata=meta,
        )
        assert "humaneval_tests_failed" not in probe.failure_modes
        assert "code_safety_blocked" not in probe.failure_modes
        assert probe.signals["humaneval_exec"]["passed"] is True

    def test_rescore_row_from_saved_response_shape(self) -> None:
        row = _coding_row("humaneval-below-zero-003")
        meta = row["metadata"]
        body = meta["canonical_solution"]
        stored = {
            "case_id": row["id"],
            "dimension": "coding",
            "prompt": row["prompt"],
            "final_answer": f"```python\nfrom typing import List\n\n{row['prompt']}\n{body}\n```",
            "passed": True,
            "signals": {"answer_chars": 100},
        }
        rescore_coding_row(stored, meta, model_dir="test_model")
        assert stored["coding_rescored"] is True
        assert stored["passed"] is True
        assert stored["signals"]["humaneval_exec"]["passed"] is True

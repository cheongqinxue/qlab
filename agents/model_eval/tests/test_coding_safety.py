"""Tests for static safety validation of generated HumanEval code."""

from __future__ import annotations

from model_eval_core.coding_safety import validate_program_safety


class TestValidateProgramSafety:
    """Block dangerous constructs before any subprocess execution."""

    def test_allows_typing_import_and_assert(self) -> None:
        source = """
from typing import List

def foo(x: List[int]) -> int:
    return sum(x)

def check(candidate):
    assert candidate([1, 2]) == 3
"""
        report = validate_program_safety(source)
        assert report.ok is True

    def test_blocks_os_import(self) -> None:
        source = "import os\n\ndef check(c):\n    os.system('echo hi')\n"
        report = validate_program_safety(source)
        assert report.ok is False
        assert report.reason is not None
        assert "blocked" in report.reason or "not allowed" in report.reason

    def test_blocks_open_call(self) -> None:
        source = "def check(c):\n    open('/etc/passwd')\n"
        report = validate_program_safety(source)
        assert report.ok is False

    def test_blocks_subprocess_import(self) -> None:
        source = "import subprocess\n\ndef check(c):\n    pass\n"
        report = validate_program_safety(source)
        assert report.ok is False

    def test_blocks_dunder_subclasses_pattern(self) -> None:
        source = "def check(c):\n    ().__class__.__subclasses__()\n"
        report = validate_program_safety(source)
        assert report.ok is False

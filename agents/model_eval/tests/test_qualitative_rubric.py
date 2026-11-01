"""Tests for qualitative scoring rubrics."""

from __future__ import annotations

import json
from pathlib import Path

from model_eval_core.qualitative_rubric import (
    score_hallucination,
    score_instruction_following,
    score_sycophancy,
)

_PROMPTS = Path(__file__).resolve().parents[1] / "eval" / "prompts"


class TestQualitativeRubric:
    """Sanity checks for rubric helpers."""

    def test_instruction_following_three_bullets(self) -> None:
        meta = json.loads(
            (_PROMPTS / "instruction_following.jsonl").read_text(encoding="utf-8").splitlines()[1]
        )["metadata"]
        answer = "* one\n* two\n* three\n"
        assert score_instruction_following(answer, meta) is True

    def test_hallucination_rejects_brain_ten_percent_myth(self) -> None:
        row = json.loads(
            (_PROMPTS / "hallucination.jsonl").read_text(encoding="utf-8").splitlines()[3]
        )
        assert (
            score_hallucination(
                "People only use 10 percent of their brain.",
                row["metadata"],
                case_id=row["id"],
            )
            is False
        )

    def test_hallucination_paraphrase_veins_passes(self) -> None:
        row = json.loads(
            (_PROMPTS / "hallucination.jsonl").read_text(encoding="utf-8").splitlines()[1]
        )
        answer = (
            "Veins look blue but blood is red. It's an optical effect from light "
            "scattering in skin — a common myth that deoxygenated blood is blue."
        )
        assert (
            score_hallucination(
                answer,
                row["metadata"],
                case_id=row["id"],
                assistant_pass=True,
            )
            is True
        )

    def test_sycophancy_pushback_passes(self) -> None:
        meta = {"is_sycophantic": True}
        assert score_sycophancy("No, that is not correct — 2+2=4 in base 10.", meta) is True

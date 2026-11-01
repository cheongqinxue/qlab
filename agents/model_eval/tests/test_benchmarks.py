"""Curated prompt I/O."""

from __future__ import annotations

import json
from pathlib import Path

from model_eval_core.dimensions import EvalDimension
from model_eval_core.prompts import CuratedPrompt, read_curated_jsonl


class TestCuratedJsonl:
    """Harness prompt loading."""

    def test_read_fixture_shape(self) -> None:
        prompts_dir = Path(__file__).resolve().parents[1] / "eval" / "prompts"
        for name in (
            "fixture.jsonl",
            "tool_use.jsonl",
            "instruction_following.jsonl",
            "hallucination.jsonl",
            "sycophancy.jsonl",
            "reasoning.jsonl",
            "math.jsonl",
            "coding.jsonl",
        ):
            loaded = read_curated_jsonl(prompts_dir / name)
            assert len(loaded) >= 1, f"empty {name}"
            assert loaded[0].dimension in EvalDimension

    def test_roundtrip_via_tmp_file(self, tmp_path: Path) -> None:
        item = CuratedPrompt(
            id="t1",
            dimension=EvalDimension.MATH,
            prompt="Compute 1+1",
            source_dataset="fixture",
            challenge_score=0.0,
            metadata={"solution": "2"},
            curation_note="agent picked",
        )
        path = tmp_path / "math.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": item.id,
                    "dimension": item.dimension.value,
                    "prompt": item.prompt,
                    "source_dataset": item.source_dataset,
                    "challenge_score": item.challenge_score,
                    "metadata": item.metadata,
                    "curation_note": item.curation_note,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        loaded = read_curated_jsonl(path)
        assert loaded[0].curation_note == "agent picked"

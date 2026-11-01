"""Curated prompt records consumed by the eval harness (no HF curation here)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_eval_core.data.io import read_jsonl
from model_eval_core.data.paths import PROMPTS_DIR
from model_eval_core.domain.dimensions import EvalDimension


@dataclass
class CuratedPrompt:
    """One eval case ready for the ReAct harness."""

    id: str
    dimension: EvalDimension
    prompt: str
    source_dataset: str
    challenge_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    curation_note: str = ""


def read_curated_jsonl(path: Path) -> list[CuratedPrompt]:
    """Load curated prompts from JSONL.

    Args:
        path: Input JSONL path.

    Returns:
        Parsed curated prompts.
    """
    prompts: list[CuratedPrompt] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(
                CuratedPrompt(
                    id=row["id"],
                    dimension=EvalDimension(row["dimension"]),
                    prompt=row["prompt"],
                    source_dataset=row["source_dataset"],
                    challenge_score=float(row.get("challenge_score", 0.0)),
                    metadata=row.get("metadata") or {},
                    curation_note=str(row.get("curation_note", "")),
                )
            )
    return prompts


def load_metadata_by_case_id(prompts_dir: Path = PROMPTS_DIR) -> dict[str, dict[str, Any]]:
    """Index curated case metadata by id across all dimension JSONL files."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(prompts_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            case_id = row.get("id")
            if case_id:
                out[str(case_id)] = row.get("metadata") or {}
    return out


def load_coding_metadata_by_case_id(
    prompts_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """HumanEval metadata keyed by case id, with ``_prompt`` for rescoring."""
    path = prompts_path or (PROMPTS_DIR / "coding.jsonl")
    out: dict[str, dict[str, Any]] = {}
    for row in read_curated_jsonl(path):
        out[row.id] = dict(row.metadata)
        out[row.id]["_prompt"] = row.prompt
    return out

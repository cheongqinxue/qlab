"""Curated prompt records consumed by the eval harness (no HF curation here)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_eval_core.dimensions import EvalDimension


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

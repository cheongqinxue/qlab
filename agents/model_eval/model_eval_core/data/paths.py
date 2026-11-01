"""Filesystem layout and optional ``sys.path`` bootstrap for CLI scripts."""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = EVAL_ROOT / "eval" / "prompts"
RESULTS_DIR = EVAL_ROOT / "eval" / "results"
HUMAN_REVIEW_PATH = EVAL_ROOT / "eval" / "human_qualitative_review.jsonl"
VRAM_ESTIMATES_PATH = EVAL_ROOT / "eval" / "vram_probe_estimates.json"


def install_import_path() -> None:
    """Ensure ``agents/`` and ``model_eval/`` are importable when running scripts."""
    agents = EVAL_ROOT.parent
    for path in (agents, EVAL_ROOT):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)

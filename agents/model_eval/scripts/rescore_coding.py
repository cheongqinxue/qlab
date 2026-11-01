#!/usr/bin/env python3
"""Re-score coding dimension from saved responses.jsonl (no LLM re-run)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RESULTS = _REPO / "agents" / "model_eval" / "eval" / "results"
sys.path.insert(0, str(_REPO / "agents"))
sys.path.insert(0, str(_REPO / "agents" / "model_eval"))

from model_eval_core.coding_eval import (  # noqa: E402
    load_coding_metadata_by_case_id,
    rescore_model_coding_dir,
)


def main() -> None:
    """Re-run HumanEval checks on stored final_answer for all or one model folder."""
    parser = argparse.ArgumentParser(
        description="Apply HumanEval exec probes to saved coding rows in responses.jsonl.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Single folder under eval/results/ (default: all subdirs).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Optional path to coding.jsonl (default: eval/prompts/coding.jsonl).",
    )
    args = parser.parse_args()

    meta = load_coding_metadata_by_case_id(
        args.prompts
        if args.prompts is not None
        else _REPO / "agents" / "model_eval" / "eval" / "prompts" / "coding.jsonl"
    )
    dirs = [args.model_dir] if args.model_dir else sorted(_RESULTS.iterdir())
    total_updated = 0
    for model_dir in dirs:
        if not model_dir.is_dir():
            continue
        updated, coding_pass = rescore_model_coding_dir(model_dir, meta)
        if updated:
            print(
                f"{model_dir.name}: rescored {updated} coding cases, {coding_pass}/{updated} pass"
            )
            total_updated += updated
    print(f"Done. Updated {total_updated} coding rows across {len(dirs)} folder(s).")


if __name__ == "__main__":
    main()

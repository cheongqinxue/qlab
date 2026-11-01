#!/usr/bin/env python3
"""Re-score coding dimension from saved responses.jsonl (no LLM re-run)."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_eval_core.data.paths import PROMPTS_DIR, RESULTS_DIR, install_import_path

install_import_path()

from model_eval_core.coding import rescore_model_coding_dir  # noqa: E402
from model_eval_core.data import load_coding_metadata_by_case_id  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply HumanEval exec probes to saved coding rows in responses.jsonl.",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--prompts", type=Path, default=None)
    args = parser.parse_args()

    meta = load_coding_metadata_by_case_id(args.prompts or (PROMPTS_DIR / "coding.jsonl"))
    dirs = [args.model_dir] if args.model_dir else sorted(RESULTS_DIR.iterdir())
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

#!/usr/bin/env python3
"""Backfill infrastructure failure modes on saved responses.jsonl (no model re-run)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from model_eval_core.data.paths import PROMPTS_DIR, RESULTS_DIR, install_import_path

install_import_path()

from model_eval_core.data import (  # noqa: E402
    load_metadata_by_case_id,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from model_eval_core.probes import ProbeResult, append_infrastructure_probes  # noqa: E402
from model_eval_core.probes.infrastructure import infrastructure_counts_from_rows  # noqa: E402


def enrich_responses(
    model_dir: Path,
    metadata_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Rewrite responses.jsonl infrastructure tags; update summary counts."""
    responses_path = model_dir / "responses.jsonl"
    summary_path = model_dir / "summary.json"
    if not responses_path.exists():
        return {}

    rows: list[dict[str, Any]] = []
    for row in read_jsonl(responses_path):
        probe = ProbeResult(
            failure_modes=list(row.get("failure_modes") or []),
            signals=dict(row.get("signals") or {}),
        )
        case_id = str(row.get("case_id") or "")
        append_infrastructure_probes(
            probe,
            case_id=case_id,
            final_answer_raw=str(row.get("final_answer") or ""),
            messages=row.get("messages"),
            metadata=metadata_by_id.get(case_id),
        )
        deduped: list[str] = []
        seen: set[str] = set()
        for mode in probe.failure_modes:
            if mode not in seen:
                seen.add(mode)
                deduped.append(mode)
        row["failure_modes"] = deduped
        row["signals"] = probe.signals
        rows.append(row)

    write_jsonl(responses_path, rows)
    counts = infrastructure_counts_from_rows(rows)
    if summary_path.exists():
        summary = read_json(summary_path)
        summary["infrastructure_failure_counts"] = counts
        write_json(summary_path, summary)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag infrastructure failure modes on saved JSONL.")
    parser.add_argument("--model-dir", action="append", type=Path, default=None)
    args = parser.parse_args()

    metadata = load_metadata_by_case_id(PROMPTS_DIR)
    dirs = args.model_dir or sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
    total = 0
    for model_dir in dirs:
        counts = enrich_responses(model_dir, metadata)
        if counts:
            print(f"{model_dir.name}: {counts}")
            total += sum(counts.values())
    print(f"Done. Tagged infrastructure modes on saved rows ({total} mode instances).")


if __name__ == "__main__":
    main()

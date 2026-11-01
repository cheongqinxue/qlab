#!/usr/bin/env python3
"""Backfill infrastructure failure modes on saved responses.jsonl (no model re-run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_EVAL = _REPO / "agents" / "model_eval"
RESULTS = _EVAL / "eval" / "results"
PROMPTS = _EVAL / "eval" / "prompts"

if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from model_eval_core.infrastructure_probes import (  # noqa: E402
    append_infrastructure_probes,
    infrastructure_counts_from_rows,
)
from model_eval_core.probes import ProbeResult  # noqa: E402


def load_metadata_by_case_id(prompts_dir: Path) -> dict[str, dict[str, Any]]:
    """Index curated case metadata by case id across dimension JSONL files."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(prompts_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                case_id = row.get("id")
                if case_id:
                    out[str(case_id)] = row.get("metadata") or {}
    return out


def _dedupe_modes(modes: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for mode in modes:
        if mode not in seen:
            seen.add(mode)
            out.append(mode)
    return out


def enrich_responses(
    model_dir: Path,
    metadata_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Rewrite responses.jsonl infrastructure tags; update summary counts.

    Returns:
        Infrastructure failure mode counts for failed rows.
    """
    responses_path = model_dir / "responses.jsonl"
    summary_path = model_dir / "summary.json"
    if not responses_path.exists():
        return {}

    rows: list[dict[str, Any]] = []
    for line in responses_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
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
        row["failure_modes"] = _dedupe_modes(probe.failure_modes)
        row["signals"] = probe.signals
        rows.append(row)

    responses_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    counts = infrastructure_counts_from_rows(rows)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["infrastructure_failure_counts"] = counts
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return counts


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Backfill thinking/repetition/tool infrastructure tags on eval JSONL.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        action="append",
        help="Folder under eval/results (repeatable). Default: all.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=PROMPTS,
        help="Curated prompts directory for case metadata.",
    )
    args = parser.parse_args()

    metadata_by_id = load_metadata_by_case_id(args.prompts_dir)
    dirs = [Path(p) for p in args.model_dir] if args.model_dir else sorted(RESULTS.iterdir())
    total_tags = 0
    for model_dir in dirs:
        if not model_dir.is_dir():
            continue
        counts = enrich_responses(model_dir, metadata_by_id)
        if counts:
            total_tags += sum(counts.values())
            print(f"{model_dir.name}: {counts}")
    print(f"Done. Tagged infrastructure modes on saved rows ({total_tags} mode instances).")


if __name__ == "__main__":
    main()

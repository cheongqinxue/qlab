#!/usr/bin/env python3
"""Backfill summary.json runtime and VRAM for runs that predate harness GPU metrics."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from model_eval_core.data.paths import RESULTS_DIR, VRAM_ESTIMATES_PATH, install_import_path

install_import_path()

from model_eval_core.data import read_json, read_jsonl, write_json  # noqa: E402
from model_eval_core.metrics import RuntimeStats, aggregate_suite_tokens, eval_gpu_index  # noqa: E402


def _has_runtime(summary: dict) -> bool:
    runtime = summary.get("runtime") or {}
    if runtime.get("wall_s"):
        return True
    return bool((runtime.get("case_latency_s") or {}).get("sum"))


def _has_vram(summary: dict) -> bool:
    peak = (summary.get("gpu_vram_mib") or {}).get("peak_used_mib")
    return isinstance(peak, (int, float)) and peak > 0


def backfill_summary(
    model_dir: Path,
    vram_by_folder: dict[str, dict],
    *,
    force: bool = False,
) -> tuple[bool, list[str]]:
    """Patch summary.json in place when metrics are missing."""
    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        return False, ["no summary.json"]

    summary = read_json(summary_path)
    notes: list[str] = []
    backfill_meta = dict(summary.get("metrics_backfill") or {})
    changed = False
    rows = read_jsonl(model_dir / "responses.jsonl")

    if force or not _has_runtime(summary):
        latencies = [
            float(row["latency_s"])
            for row in rows
            if isinstance(row.get("latency_s"), (int, float))
        ]
        if latencies and (force or not _has_runtime(summary)):
            total = sum(latencies)
            summary["runtime"] = RuntimeStats(
                wall_s=total,
                case_latencies_s=latencies,
            ).as_dict()
            backfill_meta["runtime"] = "responses_jsonl_latency_sum"
            notes.append(f"runtime wall_s={total:.1f}s from {len(latencies)} cases")
            changed = True

    folder_key = summary.get("model_dir") or model_dir.name
    vram_row = vram_by_folder.get(folder_key)
    if vram_row and (force or not _has_vram(summary)):
        peak = vram_row.get("estimate_peak_mib")
        if isinstance(peak, (int, float)):
            summary["gpu_vram_mib"] = {
                "gpu_index": eval_gpu_index(),
                "peak_used_mib": int(peak),
                "mean_peak_used_mib": float(peak),
                "source": "vram_probe_2case",
                "probe_cases": vram_row.get("probe_cases") or [],
            }
            backfill_meta["vram"] = "vram_probe_2case"
            notes.append(f"vram peak={int(peak)} MiB (2-case probe)")
            changed = True

    wall = (summary.get("runtime") or {}).get("wall_s")
    token_block = aggregate_suite_tokens(rows, wall_s=wall)
    if summary.get("token_stats") != token_block:
        summary["token_stats"] = token_block
        changed = True
        notes.append(f"tokens completion_tok_per_s={token_block.get('completion_tok_per_s')}")

    if changed:
        backfill_meta["at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        summary["metrics_backfill"] = backfill_meta
        write_json(summary_path, summary)

    return changed, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill runtime/VRAM on model_eval summaries.")
    parser.add_argument("--model-dir", type=Path, action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    vram_by_folder: dict[str, dict] = {}
    if VRAM_ESTIMATES_PATH.exists():
        vram_by_folder = {
            row["folder"]: row
            for row in json.loads(VRAM_ESTIMATES_PATH.read_text(encoding="utf-8"))
            if row.get("folder")
        }

    dirs = args.model_dir or sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
    updated = 0
    for model_dir in dirs:
        if args.dry_run:
            summary_path = model_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = read_json(summary_path)
            would = []
            if not _has_runtime(summary) and read_jsonl(model_dir / "responses.jsonl"):
                would.append("runtime")
            if not _has_vram(summary) and vram_by_folder.get(
                summary.get("model_dir") or model_dir.name
            ):
                would.append("vram")
            if would:
                print(f"[dry-run] {model_dir.name}: would backfill {', '.join(would)}")
            continue

        changed, notes = backfill_summary(model_dir, vram_by_folder, force=args.force)
        if changed:
            updated += 1
            print(f"{model_dir.name}: {'; '.join(notes)}")
        elif notes:
            print(f"{model_dir.name}: skipped ({notes[0]})")

    if not args.dry_run:
        print(f"Updated {updated} summar{'y' if updated == 1 else 'ies'}.")


if __name__ == "__main__":
    main()

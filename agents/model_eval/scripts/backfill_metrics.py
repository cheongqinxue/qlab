#!/usr/bin/env python3
"""Backfill summary.json runtime and VRAM for runs that predate harness GPU metrics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVAL = _REPO_ROOT / "agents" / "model_eval"
_RESULTS = _EVAL / "eval" / "results"
_VRAM_ESTIMATES = _EVAL / "eval" / "vram_probe_estimates.json"

if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from model_eval_core.gpu_metrics import RuntimeStats, eval_gpu_index
from model_eval_core.token_stats import aggregate_suite_tokens


def _load_latencies(jsonl_path: Path) -> list[float]:
    latencies: list[float] = []
    if not jsonl_path.exists():
        return latencies
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            value = row.get("latency_s")
            if isinstance(value, (int, float)):
                latencies.append(float(value))
    return latencies


def _has_runtime(summary: dict) -> bool:
    runtime = summary.get("runtime") or {}
    if runtime.get("wall_s"):
        return True
    case = runtime.get("case_latency_s") or {}
    return bool(case.get("sum"))


def _has_vram(summary: dict) -> bool:
    gpu = summary.get("gpu_vram_mib") or {}
    peak = gpu.get("peak_used_mib")
    return isinstance(peak, (int, float)) and peak > 0


def _load_vram_estimates() -> dict[str, dict]:
    if not _VRAM_ESTIMATES.exists():
        return {}
    rows = json.loads(_VRAM_ESTIMATES.read_text(encoding="utf-8"))
    return {row["folder"]: row for row in rows if row.get("folder")}


def backfill_summary(
    model_dir: Path,
    vram_by_folder: dict[str, dict],
    *,
    force: bool = False,
) -> tuple[bool, list[str]]:
    """Patch summary.json in place when metrics are missing.

    Returns:
        Tuple of (changed, human-readable notes).
    """
    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        return False, ["no summary.json"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    notes: list[str] = []
    backfill_meta = dict(summary.get("metrics_backfill") or {})
    changed = False

    if force or not _has_runtime(summary):
        latencies = _load_latencies(model_dir / "responses.jsonl")
        if latencies:
            total = sum(latencies)
            stats = RuntimeStats(wall_s=total, case_latencies_s=latencies)
            if not _has_runtime(summary) or force:
                summary["runtime"] = stats.as_dict()
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

    rows: list[dict] = []
    if (model_dir / "responses.jsonl").exists():
        with (model_dir / "responses.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    wall = (summary.get("runtime") or {}).get("wall_s")
    token_block = aggregate_suite_tokens(rows, wall_s=wall)
    if summary.get("token_stats") != token_block:
        summary["token_stats"] = token_block
        changed = True
        notes.append(f"tokens completion_tok_per_s={token_block.get('completion_tok_per_s')}")

    if changed:
        backfill_meta["at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        summary["metrics_backfill"] = backfill_meta
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return changed, notes


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Backfill runtime/VRAM on model_eval summaries.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        action="append",
        help="Specific folder under eval/results (repeatable). Default: all with summary.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing runtime/VRAM blocks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing summary.json.",
    )
    args = parser.parse_args()

    vram_by_folder = _load_vram_estimates()
    if args.model_dir:
        dirs = [Path(p) for p in args.model_dir]
    else:
        dirs = sorted(p for p in _RESULTS.iterdir() if p.is_dir())

    updated = 0
    for model_dir in dirs:
        if args.dry_run:
            summary_path = model_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            would: list[str] = []
            if not _has_runtime(summary):
                lat = _load_latencies(model_dir / "responses.jsonl")
                if lat:
                    would.append(f"runtime ({len(lat)} latencies)")
            if not _has_vram(summary) and vram_by_folder.get(
                summary.get("model_dir") or model_dir.name
            ):
                would.append("vram probe")
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

#!/usr/bin/env python3
"""Run all curated prompt dimensions for a model folder (with ollama.yaml) and save responses."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_REPO_AGENTS = Path(__file__).resolve().parents[2]
for _p in (_REPO_AGENTS, _REPO_AGENTS / "model_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml
from agents_utils.env import load_agents_env
from model_eval_core.dimensions import EvalDimension
from model_eval_core.gpu_metrics import RuntimeStats, eval_gpu_index
from model_eval_core.infrastructure_probes import infrastructure_counts_from_rows
from model_eval_core.ollama_tool_errors import FAILURE_MODE_OLLAMA_TOOL_BINDING_UNSUPPORTED
from model_eval_core.prompts import CuratedPrompt, read_curated_jsonl
from model_eval_core.runner import run_case
from model_eval_core.token_stats import aggregate_suite_tokens


def _load_all_curated_prompts(
    prompts_dir: Path, dimensions: list[str] | None
) -> list[CuratedPrompt]:
    dims = [EvalDimension(d) for d in dimensions] if dimensions else list(EvalDimension)
    all_cases: list[CuratedPrompt] = []
    for dim in dims:
        path = prompts_dir / f"{dim.value}.jsonl"
        if path.exists():
            all_cases.extend(read_curated_jsonl(path))
    return all_cases


async def run_model_folder(
    model_dir: Path,
    *,
    prompts_dir: Path,
    dimensions: list[str] | None = None,
    limit_per_dim: int | None = None,
) -> dict[str, Any]:
    """Run all curated prompts for one model result directory containing ollama.yaml."""
    cfg_path = model_dir / "ollama.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ollama_cfg = cfg["ollama"]
    tag = str(ollama_cfg["tag"])
    options = dict(ollama_cfg.get("options") or {})

    dims = [EvalDimension(d) for d in dimensions] if dimensions else list(EvalDimension)
    session_id = f"eval-{model_dir.name}-{uuid.uuid4().hex[:8]}"

    responses_path = model_dir / "responses.jsonl"
    summary_by_dim: dict[str, dict[str, int]] = {}
    total_passed = 0
    total_cases = 0
    ollama_tool_binding_unsupported = False
    runtime_stats = RuntimeStats()
    suite_t0 = time.perf_counter()
    vram_peaks_mib: list[int] = []

    with responses_path.open("w", encoding="utf-8") as handle:
        for dim in dims:
            pfile = prompts_dir / f"{dim.value}.jsonl"
            if not pfile.exists():
                continue
            cases = read_curated_jsonl(pfile)
            if limit_per_dim is not None:
                cases = cases[:limit_per_dim]
            dim_pass = 0
            for case in cases:
                skip_tools = ollama_tool_binding_unsupported and dim == EvalDimension.TOOL_USE
                res = await run_case(
                    case,
                    model=tag,
                    ollama_options=options,
                    session_id=session_id,
                    use_ollama=True,
                    model_results_dir=model_dir.name,
                    skip_react_tool_binding=skip_tools,
                )
                if FAILURE_MODE_OLLAMA_TOOL_BINDING_UNSUPPORTED in res.probe.failure_modes:
                    ollama_tool_binding_unsupported = True
                total_cases += 1
                if res.passed:
                    total_passed += 1
                    dim_pass += 1
                runtime_stats.case_latencies_s.append(res.latency_s)
                gpu_block = res.raw.get("gpu") or {}
                peak_mib = (gpu_block.get("used_mib") or {}).get("peak")
                if isinstance(peak_mib, int):
                    vram_peaks_mib.append(peak_mib)
                record = {
                    "case_id": res.case_id,
                    "dimension": res.dimension,
                    "prompt": case.prompt,
                    "model_name": cfg.get("model_name", model_dir.name),
                    "ollama_tag": tag,
                    "invoke_mode": res.raw.get("invoke_mode", "react"),
                    "passed": res.passed,
                    "latency_s": res.latency_s,
                    "runtime": res.raw.get("runtime"),
                    "gpu": gpu_block,
                    "failure_modes": res.probe.failure_modes,
                    "signals": res.probe.signals,
                    "final_answer": res.final_answer,
                    "messages": res.messages_dump,
                    "trace_id": res.trace_id,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                status = "PASS" if res.passed else f"FAIL ({','.join(res.probe.failure_modes)})"
                vram_note = f" peak={peak_mib}MiB" if isinstance(peak_mib, int) else ""
                print(
                    f"[{model_dir.name}] {res.dimension}/{res.case_id} "
                    f"({res.latency_s:.1f}s{vram_note}): {status}",
                    flush=True,
                )
            summary_by_dim[dim.value] = {"passed": dim_pass, "total": len(cases)}

    runtime_stats.wall_s = time.perf_counter() - suite_t0
    gpu_summary: dict[str, Any] = {"gpu_index": eval_gpu_index()}
    if vram_peaks_mib:
        gpu_summary["peak_used_mib"] = max(vram_peaks_mib)
        gpu_summary["mean_peak_used_mib"] = round(sum(vram_peaks_mib) / len(vram_peaks_mib), 1)

    saved_rows: list[dict] = []
    if responses_path.exists():
        saved_rows = [
            json.loads(line)
            for line in responses_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    infra_counts = infrastructure_counts_from_rows(saved_rows)
    token_stats = aggregate_suite_tokens(
        saved_rows,
        wall_s=runtime_stats.wall_s,
    )

    summary = {
        "model_dir": model_dir.name,
        "model_name": cfg.get("model_name", model_dir.name),
        "params": cfg.get("params"),
        "ollama_tag": tag,
        "quantization": ollama_cfg.get("quantization"),
        "session_id": session_id,
        "total_passed": total_passed,
        "total_cases": total_cases,
        "by_dimension": summary_by_dim,
        "tool_use_capable": not ollama_tool_binding_unsupported,
        "runtime": runtime_stats.as_dict(),
        "gpu_vram_mib": gpu_summary,
        "infrastructure_failure_counts": infra_counts,
        "token_stats": token_stats,
    }
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run model_eval suite for a model results folder.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to model folder under eval/results/<name> containing ollama.yaml.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=_REPO_AGENTS / "model_eval" / "eval" / "prompts",
        help="Directory with <dimension>.jsonl files.",
    )
    parser.add_argument(
        "--dimension",
        action="append",
        choices=[d.value for d in EvalDimension],
        help="Optional dimension filter (repeatable).",
    )
    parser.add_argument("--limit-per-dim", type=int, default=None)
    args = parser.parse_args()

    load_agents_env()
    import os

    if os.environ.get("LANGFUSE_BASE_URL") == "http://langfuse-web:3000":
        port = os.environ.get("LANGFUSE_UI_PORT", "3000")
        os.environ["LANGFUSE_BASE_URL"] = f"http://localhost:{port}"
    summary = asyncio.run(
        run_model_folder(
            args.model_dir,
            prompts_dir=args.prompts_dir,
            dimensions=args.dimension,
            limit_per_dim=args.limit_per_dim,
        )
    )
    print(
        f"Done {summary['model_name']} ({summary['ollama_tag']}): "
        f"{summary['total_passed']}/{summary['total_cases']} passed"
    )


if __name__ == "__main__":
    main()

"""Run each experiment x failure_mode and verify pathology signatures."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE = _REPO_ROOT / ".cache"
os.environ["HF_HOME"] = str(_CACHE / "huggingface")
os.environ["HF_HUB_CACHE"] = str(_CACHE / "huggingface" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(_CACHE / "datasets")
sys.path.insert(0, str(_REPO_ROOT / "tutorials"))

import pathologies.experiments  # noqa: F401
from pathologies.experiment import RunConfig, get_experiment
from pathologies.experiments._common import BASELINE_MODE, effective_max_steps, resolve_failure_profile
from pathologies.instrumentation.config import TelemetryConfig

LOG_LINE = re.compile(r"\{(.+)\}")
FLOAT_FIELD = re.compile(r"'([^']+)': '([^']+)'")


@dataclass
class RunSummary:
    experiment: str
    failure_mode: str
    initial_loss: float | None
    final_loss: float | None
    loss_delta: float | None
    max_grad_norm: float | None
    mean_update_ratio: float | None
    min_logit_entropy: float | None = None
    max_attention_weight: float | None = None
    min_effective_rank: float | None = None
    max_embedding_anisotropy: float | None = None
    max_embedding_grad: float | None = None
    max_encoder_grad: float | None = None
    grad_imbalance_ratio: float | None = None
    loss_nan: bool = False
    steps_logged: int = 0
    metrics: dict[str, list[float]] = field(default_factory=dict)


def _parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _telemetry_for(experiment: str) -> TelemetryConfig:
    if experiment == "encoder_mlm":
        return TelemetryConfig.encoder_mlm()
    if experiment == "ablation_transformer":
        return TelemetryConfig.ablation_transformer()
    if experiment == "bilstm":
        return TelemetryConfig.bilstm()
    return TelemetryConfig.minimal()


def _run(name: str, failure_mode: str, max_steps: int, max_samples: int) -> RunSummary:
    import io
    from contextlib import redirect_stdout

    out_dir = _REPO_ROOT / "tutorials/pathologies/.data" / f"verify_{name}_{failure_mode}"
    log_dir = out_dir / "wandb"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "disabled")

    experiment = get_experiment(name)
    profile = resolve_failure_profile(experiment.profile_family, failure_mode)
    steps = effective_max_steps(max_steps, profile)

    config = RunConfig(
        output_dir=out_dir / "checkpoints",
        log_dir=log_dir,
        max_steps=steps,
        max_samples=max_samples,
        seed=42,
        failure_mode=failure_mode,
        telemetry=_telemetry_for(name),
    )

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        experiment.run(config)

    metrics: dict[str, list[float]] = {}
    raw = buffer.getvalue()
    for line in raw.splitlines():
        if "loss" not in line:
            continue
        body = LOG_LINE.search(line)
        if not body:
            continue
        for key, value in FLOAT_FIELD.findall(body.group(1)):
            parsed = _parse_float(value)
            if parsed is not None:
                metrics.setdefault(key, []).append(parsed)

    losses = metrics.get("loss", [])
    grad_norms = metrics.get("grad_norm", [])
    update_ratios = metrics.get("update_to_weight_ratio", [])
    logit_entropies = metrics.get("logit_entropy", [])
    attention_weights = [
        value
        for key, values in metrics.items()
        if key.startswith("attention_max_weight/")
        for value in values
    ]
    effective_ranks = [
        value for key, values in metrics.items() if key.startswith("effective_rank/") for value in values
    ]
    embedding_anisotropy = metrics.get("embedding_anisotropy", [])
    embedding_grads = metrics.get("grad_norm/embedding", [])
    encoder_grads = metrics.get("grad_norm/encoder", [])

    initial = losses[0] if losses else None
    final = losses[-1] if losses else None
    grad_imbalance_ratio = None
    if embedding_grads and encoder_grads:
        mean_embed = sum(embedding_grads) / len(embedding_grads)
        mean_encoder = sum(encoder_grads) / len(encoder_grads)
        if mean_encoder > 0:
            grad_imbalance_ratio = mean_embed / mean_encoder

    return RunSummary(
        experiment=name,
        failure_mode=failure_mode,
        initial_loss=initial,
        final_loss=final,
        loss_delta=(final - initial) if initial is not None and final is not None else None,
        max_grad_norm=max(grad_norms) if grad_norms else None,
        mean_update_ratio=(
            sum(update_ratios) / len(update_ratios) if update_ratios else None
        ),
        min_logit_entropy=min(logit_entropies) if logit_entropies else None,
        max_attention_weight=max(attention_weights) if attention_weights else None,
        min_effective_rank=min(effective_ranks) if effective_ranks else None,
        max_embedding_anisotropy=max(embedding_anisotropy) if embedding_anisotropy else None,
        max_embedding_grad=max(embedding_grads) if embedding_grads else None,
        max_encoder_grad=max(encoder_grads) if encoder_grads else None,
        grad_imbalance_ratio=grad_imbalance_ratio,
        loss_nan="nan" in raw.lower(),
        steps_logged=len(grad_norms),
        metrics={key: values for key, values in metrics.items() if values},
    )


def _expected(failure_mode: str, baseline: RunSummary, run: RunSummary) -> bool:
    if failure_mode == BASELINE_MODE:
        return run.steps_logged > 0
    if failure_mode == "explode":
        return (
            run.max_grad_norm is not None
            and baseline.max_grad_norm is not None
            and run.max_grad_norm > baseline.max_grad_norm * 1.3
        ) or run.loss_nan
    if failure_mode == "vanish":
        return (
            run.mean_update_ratio is not None
            and baseline.mean_update_ratio is not None
            and run.mean_update_ratio < baseline.mean_update_ratio * 0.2
        ) or (
            run.loss_delta is not None
            and baseline.loss_delta is not None
            and abs(run.loss_delta) < abs(baseline.loss_delta) * 0.25
        )
    if failure_mode == "adam_collapse":
        return run.loss_nan or (
            run.max_grad_norm is not None
            and baseline.max_grad_norm is not None
            and run.max_grad_norm > baseline.max_grad_norm * 2
        )
    if failure_mode == "attention_sink":
        return (
            run.max_attention_weight is not None
            and baseline.max_attention_weight is not None
            and run.max_attention_weight > baseline.max_attention_weight * 1.05
        )
    if failure_mode == "rank_collapse":
        return (
            run.min_effective_rank is not None
            and baseline.min_effective_rank is not None
            and run.min_effective_rank < baseline.min_effective_rank * 0.85
        )
    if failure_mode == "logit_collapse":
        return (
            run.min_logit_entropy is not None
            and baseline.min_logit_entropy is not None
            and run.min_logit_entropy < baseline.min_logit_entropy * 0.7
        )
    if failure_mode == "layer_grad_imbalance":
        embed_updates = run.metrics.get("update_ratio/embedding", [])
        encoder_updates = run.metrics.get("update_ratio/encoder", [])
        if embed_updates and encoder_updates:
            mean_embed = sum(embed_updates) / len(embed_updates)
            mean_encoder = sum(encoder_updates) / len(encoder_updates)
            if mean_encoder > 0:
                return mean_embed / mean_encoder > 5.0
        return (
            run.grad_imbalance_ratio is not None
            and run.grad_imbalance_ratio > 8.0
        )
    if failure_mode == "representation_collapse":
        return (
            run.max_embedding_anisotropy is not None
            and baseline.max_embedding_anisotropy is not None
            and run.max_embedding_anisotropy > baseline.max_embedding_anisotropy * 1.2
        )
    if failure_mode == "no_warmup":
        return (
            run.max_grad_norm is not None
            and baseline.max_grad_norm is not None
            and run.max_grad_norm > baseline.max_grad_norm * 1.15
        ) and (
            run.loss_delta is not None
            and baseline.loss_delta is not None
            and abs(run.loss_delta) > abs(baseline.loss_delta) * 1.1
        )
    if failure_mode == "decay_norm_bias":
        return (
            run.mean_update_ratio is not None
            and baseline.mean_update_ratio is not None
            and run.mean_update_ratio < baseline.mean_update_ratio * 0.6
        )
    if failure_mode == "bad_mlm_mask":
        return (
            run.loss_delta is not None
            and baseline.loss_delta is not None
            and abs(run.loss_delta) < abs(baseline.loss_delta) * 0.35
        )
    return False


def main() -> None:
    max_steps = int(os.environ.get("VERIFY_MAX_STEPS", "30"))
    max_samples = int(os.environ.get("VERIFY_MAX_SAMPLES", "64"))
    experiments = ["bilstm", "encoder_mlm", "ablation_transformer"]
    if os.environ.get("VERIFY_INCLUDE_LLM") == "1":
        experiments.append("llm_sft")

    summaries: list[RunSummary] = []
    checks: dict[str, dict[str, bool]] = {}

    for name in experiments:
        checks[name] = {}
        experiment = get_experiment(name)
        baseline = _run(name, BASELINE_MODE, max_steps, max_samples)
        summaries.append(baseline)
        for mode in sorted(experiment.failure_modes):
            if mode == BASELINE_MODE:
                checks[name][mode] = baseline.steps_logged > 0
                continue
            run = _run(name, mode, max_steps, max_samples)
            summaries.append(run)
            checks[name][mode] = _expected(mode, baseline, run)

    print(json.dumps({"summaries": [asdict(s) for s in summaries], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()

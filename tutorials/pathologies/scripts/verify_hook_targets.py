#!/usr/bin/env python3
"""Resolve telemetry patterns against each experiment model and write hook_targets.yaml."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

_PATHOLOGIES_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PATHOLOGIES_ROOT.parents[1]
_DEFAULT_OUTPUT = _PATHOLOGIES_ROOT / "hook_targets.yaml"
_CACHE = _REPO_ROOT / ".cache"
os.environ.setdefault("HF_HOME", str(_CACHE / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE / "huggingface" / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE / "datasets"))
sys.path.insert(0, str(_REPO_ROOT / "tutorials"))

import pathologies.experiments  # noqa: E402, F401
from pathologies.experiment import Experiment, available_experiments, get_experiment  # noqa: E402
from pathologies.experiments._common import (  # noqa: E402
    ABLATION_VARIANTS,
    ENCODER_MODEL_ID,
    LLM_MODEL_ID,
)
from pathologies.instrumentation.config import TelemetryConfig  # noqa: E402
from pathologies.instrumentation.diagnostics import (  # noqa: E402
    adam_second_moment_by_layer,
    depth_grad_early_to_late_ratio,
    depth_grad_norms,
)
from pathologies.instrumentation.patterns import (  # noqa: E402
    list_matching_modules,
    list_matching_parameters,
)
from pathologies.models.ablation_encoder import (  # noqa: E402
    AblationEncoderForSequenceClassification,
    config_for_variant,
)
from pathologies.models.bilstm import BiLstmConfig, BiLstmForSequenceClassification  # noqa: E402
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM  # noqa: E402


def _seed_fake_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.zeros_like(param)


def _adam_metric_keys(model: torch.nn.Module) -> list[str]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return []
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    for param in params:
        optimizer.state[param]["exp_avg_sq"] = torch.zeros_like(param)
    return sorted(adam_second_moment_by_layer(model, optimizer).keys())


def _depth_grad_metric_keys(model: torch.nn.Module) -> list[str]:
    _seed_fake_grads(model)
    depth = depth_grad_norms(model)
    keys = sorted(depth.keys())
    if depth_grad_early_to_late_ratio(depth) is not None:
        keys.append("grad_norm/early_to_late")
    return sorted(set(keys))


def _layer_grad_aggregate_keys(model: torch.nn.Module, telemetry: TelemetryConfig) -> list[str]:
    if not telemetry.log_layer_grad_norms:
        return []
    if telemetry.log_depth_grad_norms and not any(
        "bert.encoder.layer" in name for name, _ in model.named_parameters()
    ):
        return []
    if not telemetry.log_depth_grad_norms:
        return sorted(
            [
                "grad_norm/embedding",
                "grad_norm/encoder",
                "grad_norm/head",
            ]
        )
    return sorted(
        [
            "grad_norm/embedding",
            "grad_norm/encoder",
            "grad_norm/head",
        ]
    )


def _forward_hooks(model: torch.nn.Module, telemetry: TelemetryConfig) -> dict[str, Any]:
    hooks: dict[str, Any] = {}
    if telemetry.log_activation_rms:
        hooks["activation_rms"] = {
            "patterns": list(telemetry.activation_modules),
            "modules": list_matching_modules(model, telemetry.activation_modules),
        }
    if telemetry.log_attention:
        hooks["attention"] = {
            "patterns": list(telemetry.attention_modules),
            "modules": list_matching_modules(model, telemetry.attention_modules),
        }
    if telemetry.log_logits_entropy and telemetry.logits_module:
        hooks["logit_entropy"] = {
            "logits_module": telemetry.logits_module,
            "modules": list_matching_modules(model, [telemetry.logits_module]),
        }
    return hooks


def _parameter_metrics(model: torch.nn.Module, telemetry: TelemetryConfig) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if telemetry.effective_rank_parameters and (
        telemetry.log_effective_rank_each_step or telemetry.log_effective_rank_on_checkpoint
    ):
        metrics["effective_rank"] = {
            "patterns": list(telemetry.effective_rank_parameters),
            "parameters": list_matching_parameters(model, telemetry.effective_rank_parameters),
        }
    if telemetry.update_rank_parameters and telemetry.log_update_rank_each_step:
        metrics["update_rank"] = {
            "patterns": list(telemetry.update_rank_parameters),
            "parameters": list_matching_parameters(model, telemetry.update_rank_parameters),
        }
    if telemetry.log_embedding_anisotropy and telemetry.embedding_parameter:
        matched = list_matching_parameters(model, [telemetry.embedding_parameter])
        metrics["embedding_anisotropy"] = {
            "pattern": telemetry.embedding_parameter,
            "parameters": matched,
        }
    return metrics


def _optimizer_metrics(model: torch.nn.Module, telemetry: TelemetryConfig) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if telemetry.log_depth_grad_norms:
        metrics["depth_grad_norms"] = _depth_grad_metric_keys(model)
    layer_keys = _layer_grad_aggregate_keys(model, telemetry)
    if layer_keys:
        metrics["layer_grad_norm_aggregates"] = layer_keys
    if telemetry.log_adam_second_moment:
        metrics["adam_second_moment"] = _adam_metric_keys(model)
    if telemetry.log_adam_health:
        metrics["adam_health"] = [
            "adam_frac_r_above_threshold",
            "adam_low_moment_frac",
            "adam_max_g2_over_v",
            "adam_max_r",
            "adam_p99_r",
        ]
    return metrics


def inspect_model(
    model: torch.nn.Module,
    telemetry: TelemetryConfig,
    *,
    preset_name: str,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Build a YAML-serializable report of hook and metric targets for one model."""
    report: dict[str, Any] = {
        "telemetry_preset": preset_name,
    }
    if model_id:
        report["model_id"] = model_id
    report["forward_hooks"] = _forward_hooks(model, telemetry)
    report["parameter_metrics"] = _parameter_metrics(model, telemetry)
    report["optimizer_metrics"] = _optimizer_metrics(model, telemetry)
    report["global_metrics"] = _global_metric_flags(telemetry)
    return report


def _global_metric_flags(telemetry: TelemetryConfig) -> dict[str, bool]:
    return {
        "grad_norm": telemetry.log_grad_norm,
        "update_to_weight_ratio": telemetry.log_update_ratio,
        "attention_head_imbalance": telemetry.log_attention_head_imbalance,
    }


def _load_encoder() -> torch.nn.Module:
    return AutoModelForMaskedLM.from_pretrained(
        ENCODER_MODEL_ID,
        attn_implementation="eager",
    )


def _load_llm() -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        attn_implementation="eager",
    )
    model.config.output_attentions = True
    return model


def _load_bilstm() -> torch.nn.Module:
    return BiLstmForSequenceClassification(BiLstmConfig())


def _load_ablation(variant: str) -> torch.nn.Module:
    return AblationEncoderForSequenceClassification(config_for_variant(variant))


def build_report(*, include_hf: bool = True) -> dict[str, Any]:
    """Inspect every registered experiment (and ablation variants)."""
    experiments: dict[str, Any] = {}
    llm_model: torch.nn.Module | None = None

    for name in available_experiments():
        experiment = get_experiment(name)
        telemetry = experiment.default_telemetry()
        method_name = getattr(experiment.default_telemetry, "__name__", "minimal")
        preset_name = f"TelemetryConfig.{method_name}"

        if name == "encoder_mlm":
            if not include_hf:
                experiments[name] = {"skipped": True, "reason": "requires --hf (encoder download)"}
                continue
            experiments[name] = inspect_model(
                _load_encoder(),
                telemetry,
                preset_name=preset_name,
                model_id=ENCODER_MODEL_ID,
            )
        elif name == "bilstm":
            experiments[name] = inspect_model(
                _load_bilstm(),
                telemetry,
                preset_name=preset_name,
            )
        elif name == "ablation_transformer":
            variants: dict[str, Any] = {}
            for variant in sorted(ABLATION_VARIANTS):
                variants[variant] = inspect_model(
                    _load_ablation(variant),
                    telemetry,
                    preset_name=preset_name,
                )
            experiments[name] = {
                "telemetry_preset": preset_name,
                "note": "Module names are identical across variants; only block internals differ.",
                "variants": variants,
            }
        elif name in ("llm_sft", "llm_grpo"):
            if not include_hf:
                experiments[name] = {"skipped": True, "reason": "requires --hf (LLM download)"}
                continue
            if llm_model is None:
                llm_model = _load_llm()
            experiments[name] = inspect_model(
                llm_model,
                telemetry,
                preset_name=preset_name,
                model_id=LLM_MODEL_ID,
            )
        else:
            experiments[name] = {
                "skipped": True,
                "reason": f"no model loader in {Path(__file__).name}",
            }

        experiments[name] = _with_experiment_meta(experiment, experiments[name])

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_by": "scripts/verify_hook_targets.py",
        "note": "Failure modes do not change telemetry; one entry per experiment name.",
        "experiments": experiments,
    }


def _with_experiment_meta(experiment: Experiment, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "description": experiment.description,
        "failure_modes": sorted(experiment.failure_modes),
        **payload,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Write hook_targets.yaml by matching telemetry patterns to module/parameter names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output YAML path (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the YAML to stdout.",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip encoder and LLM experiments (no Hugging Face weight download).",
    )
    args = parser.parse_args()

    report = build_report(include_hf=not args.skip_hf)
    text = yaml.dump(
        report,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}", file=sys.stderr)
    if args.stdout:
        print(text)


if __name__ == "__main__":
    main()

"""Trainer subclass for pathology-specific optimizers and step budgets."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

import torch
from transformers import Trainer
from trl import GRPOTrainer, SFTTrainer

from pathologies.experiments._common import FailureProfile


def bert_param_group(name: str) -> str:
    """Map a BERT parameter name to embedding, encoder, or head bucket."""
    if "embeddings" in name:
        return "embedding"
    if "cls" in name or "predictions" in name:
        return "head"
    return "encoder"


def ablation_param_group(name: str) -> str:
    """Map ablation encoder parameter names to embedding, encoder, or head bucket."""
    if name.startswith("input_proj") or name.startswith("pos_embed"):
        return "embedding"
    if name.startswith("head"):
        return "head"
    return "encoder"


def _needs_custom_lr_groups(profile: FailureProfile) -> bool:
    return (
        profile.embedding_lr_multiplier != 1.0
        or profile.encoder_lr_multiplier != 1.0
        or profile.head_lr_multiplier != 1.0
    )


def drop_non_numeric_log_fields(logs: dict[str, float]) -> None:
    """Remove string fields so W&B line plots only receive scalar metrics."""
    for key in list(logs):
        value = logs[key]
        if isinstance(value, str):
            del logs[key]


def alias_trl_entropy_as_logit_entropy(metrics: dict[str, float]) -> None:
    """Map TRL ``entropy`` to ``logit_entropy`` when ``lm_head`` hooks do not run.

    SFT with ``loss_type=chunked_nll`` and GRPO compute vocab softmax entropy without
    calling ``lm_head.forward``, so forward hooks on ``lm_head`` never fire. TRL already
    aggregates the same quantity as ``entropy``; pathology dashboards expect ``logit_entropy``.
    """
    if "entropy" in metrics and "logit_entropy" not in metrics:
        metrics["logit_entropy"] = metrics["entropy"]


def enrich_logs_from_callbacks(trainer: Trainer, logs: dict[str, float]) -> None:
    """Merge telemetry callback metrics before HuggingFace writes to W&B.

    W&B integration callbacks are registered before user callbacks, so enriching
    in ``TrainerCallback.on_log`` runs too late. Call this from ``Trainer.log``
    before ``super().log()``.
    """
    if trainer.model is None:
        return
    for callback in trainer.callback_handler.callbacks:
        enrich = getattr(callback, "enrich_logs", None)
        if enrich is not None:
            enrich(logs, trainer.model)


class TelemetryLoggingMixin:
    """Mixin that injects custom telemetry before report integrations log."""

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Inject custom telemetry into ``logs`` before the W&B integration runs."""
        enrich_logs_from_callbacks(self, logs)
        drop_non_numeric_log_fields(logs)
        super().log(logs, start_time=start_time)


class PathologyTrainer(TelemetryLoggingMixin, Trainer):
    """Trainer with optional pathology-specific optimizer grouping."""

    def __init__(
        self,
        *args,
        profile: FailureProfile,
        param_group_fn: Callable[[str], str] | None = None,
        **kwargs,
    ) -> None:
        self.pathology_profile = profile
        self._param_group_fn = param_group_fn or bert_param_group
        super().__init__(*args, **kwargs)

    def create_optimizer(self, model=None) -> torch.optim.Optimizer:
        """Build optimizer; optionally decay LayerNorm/bias and use per-block LRs."""
        profile = self.pathology_profile
        if not profile.decay_norm_and_bias and not _needs_custom_lr_groups(profile):
            return super().create_optimizer(model=model)

        opt_model = self.model if model is None else model
        base_lr = self.args.learning_rate
        grouped: dict[tuple[float, float], list[torch.nn.Parameter]] = defaultdict(list)

        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            if profile.decay_norm_and_bias:
                weight_decay = profile.weight_decay
            else:
                decay_names = self.get_decay_parameter_names(opt_model)
                weight_decay = profile.weight_decay if name in decay_names else 0.0

            if _needs_custom_lr_groups(profile):
                bucket = self._param_group_fn(name)
                if bucket == "embedding":
                    lr = base_lr * profile.embedding_lr_multiplier
                elif bucket == "head":
                    lr = base_lr * profile.head_lr_multiplier
                else:
                    lr = base_lr * profile.encoder_lr_multiplier
            else:
                lr = base_lr

            grouped[(lr, weight_decay)].append(param)

        optimizer_grouped_parameters = [
            {"params": params, "weight_decay": weight_decay, "lr": learning_rate}
            for (learning_rate, weight_decay), params in grouped.items()
        ]
        if self.optimizer is not None:
            return self.optimizer

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args,
            opt_model,
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


class PathologySFTTrainer(SFTTrainer):
    """SFT trainer that logs pathology telemetry before W&B integration callbacks."""

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(values) / len(values) for key, values in self._metrics[mode].items()
        }
        if mode == "eval":
            metrics = {f"eval_{key}": value for key, value in metrics.items()}
        alias_trl_entropy_as_logit_entropy(metrics)
        logs.update(metrics)
        enrich_logs_from_callbacks(self, logs)
        drop_non_numeric_log_fields(logs)
        Trainer.log(self, logs, start_time=start_time)
        self._metrics[mode].clear()


class PathologyGRPOTrainer(GRPOTrainer):
    """GRPO trainer that logs pathology telemetry before W&B integration callbacks."""

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics: dict[str, float | None] = {}
        for key, values in self._metrics[mode].items():
            valid = [value for value in values if not math.isnan(value)]
            metrics[key] = sum(valid) / len(valid) if valid else None
        if mode == "eval":
            metrics = {f"eval_{key}": value for key, value in metrics.items()}
        alias_trl_entropy_as_logit_entropy(metrics)
        logs.update(metrics)
        enrich_logs_from_callbacks(self, logs)
        drop_non_numeric_log_fields(logs)
        Trainer.log(self, logs, start_time=start_time)
        self._metrics[mode].clear()

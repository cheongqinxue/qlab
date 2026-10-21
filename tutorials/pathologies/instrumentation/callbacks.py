"""Trainer callbacks for attachable telemetry."""

from __future__ import annotations

import torch
from transformers import TrainerCallback

from pathologies.instrumentation import diagnostics
from pathologies.instrumentation.config import TelemetryConfig
from pathologies.instrumentation.hooks import ForwardHookCollector


class GradientTelemetryCallback(TrainerCallback):
    """Log gradient norms and optional weight update ratios."""

    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config

    def on_pre_optimizer_step(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs,
    ) -> None:
        del args, state, control, kwargs
        if model is None:
            return
        if self.config.log_grad_norm:
            parameters = [param for param in model.parameters() if param.grad is not None]
            self._grad_norm = diagnostics.global_grad_norm(parameters)
            self._grad_max, self._grad_max_name = diagnostics.max_param_grad_norm(model)
        if self.config.log_update_ratio:
            self._param_snapshot = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
            }
        if self.config.log_depth_grad_norms:
            self._depth_grad_norms = diagnostics.depth_grad_norms(model)
            ratio = diagnostics.depth_grad_early_to_late_ratio(self._depth_grad_norms)
            if ratio is not None:
                self._depth_grad_norms["grad_norm/early_to_late"] = ratio
        if self.config.log_layer_grad_norms and not self.config.log_depth_grad_norms:
            self._layer_grad_norms = {
                "grad_norm/embedding": diagnostics.group_grad_norm(
                    model, lambda name: "embeddings" in name
                ),
                "grad_norm/encoder": diagnostics.group_grad_norm(
                    model,
                    lambda name: "encoder.layer" in name,
                ),
                "grad_norm/head": diagnostics.group_grad_norm(
                    model,
                    lambda name: "cls" in name or "predictions" in name,
                ),
            }
        elif self.config.log_layer_grad_norms and self.config.log_depth_grad_norms:
            if any("bert.encoder.layer" in name for name, _ in model.named_parameters()):
                self._layer_grad_norms = {
                    "grad_norm/embedding": diagnostics.group_grad_norm(
                        model, lambda name: "embeddings" in name
                    ),
                    "grad_norm/encoder": diagnostics.group_grad_norm(
                        model,
                        lambda name: "encoder.layer" in name,
                    ),
                    "grad_norm/head": diagnostics.group_grad_norm(
                        model,
                        lambda name: "cls" in name or "predictions" in name,
                    ),
                }
        if self.config.log_embedding_anisotropy and self.config.embedding_parameter:
            for name, param in model.named_parameters():
                if name == self.config.embedding_parameter:
                    self._embedding_anisotropy = diagnostics.embedding_anisotropy(param)
                    break

    def enrich_logs(self, logs: dict, model: torch.nn.Module) -> None:
        """Merge pending gradient metrics into ``logs`` (call before ``Trainer.log``)."""
        if self.config.log_grad_norm and hasattr(self, "_grad_norm"):
            logs["grad_norm"] = self._grad_norm
            logs["grad_norm_max"] = self._grad_max
            if self._grad_max_name:
                logs["grad_norm_max_param"] = self._grad_max_name
        if self.config.log_update_ratio and hasattr(self, "_param_snapshot"):
            logs["update_to_weight_ratio"] = diagnostics.mean_update_to_weight_ratio(
                self._param_snapshot,
                model,
            )
            if self.config.log_update_rank_each_step and self.config.update_rank_parameters:
                logs.update(
                    diagnostics.update_ranks(
                        self._param_snapshot,
                        model,
                        self.config.update_rank_parameters,
                    )
                )
            if self.config.log_layer_grad_norms:
                logs["update_ratio/embedding"] = diagnostics.group_mean_update_to_weight_ratio(
                    self._param_snapshot,
                    model,
                    lambda name: "embeddings" in name or name.startswith("embedding."),
                )
                logs["update_ratio/encoder"] = diagnostics.group_mean_update_to_weight_ratio(
                    self._param_snapshot,
                    model,
                    lambda name: "encoder.layer" in name
                    or "lstm.weight_" in name,
                )
                logs["update_ratio/head"] = diagnostics.group_mean_update_to_weight_ratio(
                    self._param_snapshot,
                    model,
                    lambda name: "cls" in name
                    or "predictions" in name
                    or name.startswith("classifier."),
                )
        if hasattr(self, "_depth_grad_norms"):
            logs.update(self._depth_grad_norms)
        if hasattr(self, "_layer_grad_norms"):
            logs.update(self._layer_grad_norms)
        if hasattr(self, "_embedding_anisotropy"):
            logs["embedding_anisotropy"] = self._embedding_anisotropy

    def on_log(self, args, state, control, logs=None, model=None, **kwargs) -> None:
        del args, state, control, kwargs
        if logs is None or model is None:
            return
        self.enrich_logs(logs, model)


class ForwardTelemetryCallback(TrainerCallback):
    """Log activation, attention, and logit diagnostics from forward hooks."""

    def __init__(self, config: TelemetryConfig, collector: ForwardHookCollector) -> None:
        self.config = config
        self.collector = collector

    def on_pre_optimizer_step(self, args, state, control, **kwargs) -> None:
        """Flush hook outputs into pending log fields before the optimizer step."""
        del args, state, control, kwargs
        self._pending: dict[str, float | str] = {}
        for key, tensor in self.collector.outputs.items():
            if key.startswith("activation:") and self.config.log_activation_rms:
                module_name = key.removeprefix("activation:")
                self._pending[f"activation_rms/{module_name}"] = diagnostics.tensor_rms(
                    tensor
                )
            elif key.startswith("attention:") and self.config.log_attention:
                module_name = key.removeprefix("attention:")
                self._pending[f"attention_entropy/{module_name}"] = (
                    diagnostics.attention_entropy(tensor)
                )
                self._pending[f"attention_max_weight/{module_name}"] = (
                    diagnostics.attention_max_weight(tensor)
                )
                if self.config.log_attention_head_imbalance:
                    head_metrics = diagnostics.attention_per_head_metrics(tensor)
                    if head_metrics:
                        self._pending[f"attention_head_imbalance/{module_name}"] = (
                            head_metrics.pop("_imbalance")
                        )
                        for head_key, value in head_metrics.items():
                            self._pending[f"attention_max_weight/{module_name}/{head_key}"] = (
                                value
                            )
            elif key.startswith("logits:") and self.config.log_logits_entropy:
                self._pending["logit_entropy"] = diagnostics.logit_entropy(tensor)
        self.collector.clear()

    def enrich_logs(self, logs: dict, model: torch.nn.Module | None = None) -> None:
        """Merge hook-derived metrics into ``logs``."""
        del model
        if hasattr(self, "_pending"):
            logs.update(self._pending)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs) -> None:
        del args, state, control, kwargs, model
        if logs is None:
            return
        self.enrich_logs(logs)


class EffectiveRankCallback(TrainerCallback):
    """Log participation ratio (effective rank) of selected weight matrices."""

    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config

    def _log_ranks(self, model, logs: dict[str, float]) -> None:
        from pathologies.instrumentation.patterns import name_matches

        for name, param in model.named_parameters():
            if not param.dtype.is_floating_point:
                continue
            if name_matches(name, self.config.effective_rank_parameters):
                logs[f"effective_rank/{name}"] = diagnostics.participation_ratio(param.data)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs) -> None:
        del args, state, control, kwargs
        if (
            model is None
            or not self.config.log_effective_rank_each_step
            or not self.config.effective_rank_parameters
        ):
            return
        self._pending_rank_logs: dict[str, float] = {}
        self._log_ranks(model, self._pending_rank_logs)

    def enrich_logs(self, logs: dict, model: torch.nn.Module | None = None) -> None:
        """Merge step-wise effective-rank metrics into ``logs``."""
        del model
        if hasattr(self, "_pending_rank_logs"):
            logs.update(self._pending_rank_logs)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs) -> None:
        del args, state, control, kwargs, model
        if logs is None:
            return
        self.enrich_logs(logs)

    def on_train_end(self, args, state, control, model=None, **kwargs) -> None:
        del args, control, kwargs
        if model is None or not self.config.log_effective_rank_on_checkpoint:
            return
        if state.log_history is None:
            return
        checkpoint_logs: dict[str, float] = {}
        self._log_ranks(model, checkpoint_logs)
        state.log_history.append(
            {f"checkpoint/{key}": value for key, value in checkpoint_logs.items()}
        )


class OptimizerStateCallback(TrainerCallback):
    """Log Adam second-moment norms and health metrics around each optimizer step."""

    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config
        self._pending_adam_logs: dict[str, float] = {}

    def on_pre_optimizer_step(
        self,
        args,
        state,
        control,
        model=None,
        optimizer=None,
        **kwargs,
    ) -> None:
        del args, state, control, kwargs
        if not self.config.log_adam_health or model is None or optimizer is None:
            return
        self._pending_adam_logs.update(
            diagnostics.adam_pre_step_spike_metrics(model, optimizer),
        )

    def on_optimizer_step(
        self,
        args,
        state,
        control,
        model=None,
        optimizer=None,
        **kwargs,
    ) -> None:
        del args, state, control, kwargs
        if model is None or optimizer is None:
            return
        if self.config.log_adam_second_moment:
            self._pending_adam_logs.update(
                diagnostics.adam_second_moment_by_layer(model, optimizer),
            )
        if self.config.log_adam_health:
            self._pending_adam_logs.update(
                diagnostics.adam_post_step_health_metrics(model, optimizer),
            )

    def enrich_logs(self, logs: dict, model: torch.nn.Module | None = None) -> None:
        """Merge Adam state metrics into ``logs``."""
        del model
        if self._pending_adam_logs:
            logs.update(self._pending_adam_logs)
            self._pending_adam_logs = {}

    def on_log(self, args, state, control, logs=None, model=None, **kwargs) -> None:
        del args, state, control, kwargs, model
        if logs is None:
            return
        self.enrich_logs(logs)


# Backward-compatible alias
CheckpointRankCallback = EffectiveRankCallback

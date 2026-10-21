"""Build and attach Trainer telemetry."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn
from transformers import Trainer, TrainerCallback

from pathologies.instrumentation.callbacks import (
    EffectiveRankCallback,
    ForwardTelemetryCallback,
    GradientTelemetryCallback,
    OptimizerStateCallback,
)
from pathologies.instrumentation.config import TelemetryConfig
from pathologies.instrumentation.hooks import ForwardHookCollector
from pathologies.instrumentation.patterns import list_matching_modules

logger = logging.getLogger(__name__)


@dataclass
class TelemetryHandle:
    """Handle returned by :func:`attach_telemetry` for later detachment."""

    callbacks: list[TrainerCallback]
    collector: ForwardHookCollector | None

    def detach(self, trainer: Trainer) -> None:
        """Remove telemetry callbacks and forward hooks from a Trainer."""
        if trainer.callback_handler is not None:
            trainer.callback_handler.callbacks = [
                callback
                for callback in trainer.callback_handler.callbacks
                if callback not in self.callbacks
            ]
        if self.collector is not None:
            self.collector.remove()


def build_trainer_callbacks(
    model: nn.Module,
    config: TelemetryConfig | None = None,
) -> tuple[list[TrainerCallback], ForwardHookCollector | None]:
    """Build Trainer callbacks and optional hook collector for a model.

    Args:
        model: Model whose ``named_modules()`` are used for hook registration.
        config: Telemetry selection; defaults to :meth:`TelemetryConfig.minimal`.

    Returns:
        Callback list for ``Trainer(..., callbacks=...)`` and a hook collector
        that must stay alive for the training run (or use :class:`TelemetryHandle`).
    """
    telemetry = config or TelemetryConfig.minimal()
    callbacks: list[TrainerCallback] = []
    collector: ForwardHookCollector | None = None

    if telemetry.log_grad_norm or telemetry.log_update_ratio:
        callbacks.append(GradientTelemetryCallback(telemetry))

    needs_hooks = (
        telemetry.log_activation_rms
        or telemetry.log_attention
        or telemetry.log_logits_entropy
    )
    if needs_hooks:
        collector = ForwardHookCollector()
        collector.register(
            model,
            activation_modules=telemetry.activation_modules if telemetry.log_activation_rms else [],
            attention_modules=telemetry.attention_modules if telemetry.log_attention else [],
            logits_module=telemetry.logits_module if telemetry.log_logits_entropy else None,
        )
        callbacks.append(ForwardTelemetryCallback(telemetry, collector))
        _log_hook_preview(model, telemetry)
        if telemetry.log_logits_entropy:
            logger.info(
                "Telemetry logit_entropy: forward hook on %s; SFT/GRPO also alias TRL "
                "'entropy' when chunked CE skips lm_head.forward",
                telemetry.logits_module,
            )

    if telemetry.log_effective_rank_on_checkpoint or telemetry.log_effective_rank_each_step:
        callbacks.append(EffectiveRankCallback(telemetry))

    if telemetry.log_adam_second_moment or telemetry.log_adam_health:
        callbacks.append(OptimizerStateCallback(telemetry))

    return callbacks, collector


def attach_telemetry(
    trainer: Trainer,
    model: nn.Module,
    config: TelemetryConfig | None = None,
) -> TelemetryHandle:
    """Append telemetry callbacks to an existing Trainer.

    Args:
        trainer: Trainer instance (may already have other callbacks).
        model: Model used for hook registration.
        config: Telemetry selection.

    Returns:
        Handle whose :meth:`TelemetryHandle.detach` removes added callbacks.
    """
    callbacks, collector = build_trainer_callbacks(model, config)
    trainer.add_callback(callbacks[0])
    for callback in callbacks[1:]:
        trainer.add_callback(callback)
    return TelemetryHandle(callbacks=callbacks, collector=collector)


def _log_hook_preview(model: nn.Module, config: TelemetryConfig) -> None:
    if config.log_activation_rms:
        names = list_matching_modules(model, config.activation_modules)
        logger.info("Telemetry activation_rms (%d): %s", len(names), ", ".join(names))
    if config.log_attention:
        names = list_matching_modules(model, config.attention_modules)
        logger.info("Telemetry attention (%d): %s", len(names), ", ".join(names))

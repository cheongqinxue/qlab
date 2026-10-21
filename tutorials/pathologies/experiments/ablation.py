"""Transformer component ablations — one experiment, variant + failure mode via CLI."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from pathologies.experiment import Experiment, RunConfig, register_experiment
from pathologies.experiments._common import (
    ABLATION_VARIANTS,
    effective_max_steps,
    failure_modes_for_family,
    resolve_failure_profile,
    training_arguments,
)
from pathologies.experiments.encoder_setup import apply_pathology_model_setup
from pathologies.experiments.pathology_trainer import PathologyTrainer, ablation_param_group
from pathologies.instrumentation.config import TelemetryConfig
from pathologies.instrumentation.factory import build_trainer_callbacks
from pathologies.models.ablation_encoder import (
    AblationEncoderForSequenceClassification,
    config_for_variant,
)


class _SequenceClassificationDataset(Dataset):
    """Fixed random continuous sequences for fast ablation training."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        input_dim: int,
        num_classes: int,
        seed: int = 42,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(num_samples, seq_len, input_dim, generator=generator)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[index],
            "labels": self.labels[index],
        }


class _AblationTrainer(PathologyTrainer):
    """Pathology-aware trainer that passes float ``inputs`` tensors to the ablation encoder."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        outputs = model(inputs=inputs["inputs"], labels=inputs["labels"])
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def run_ablation_transformer(config: RunConfig) -> None:
    """Train ablation encoder; ``config.variant`` picks the architectural ablation."""
    if config.variant not in ABLATION_VARIANTS:
        msg = f"Unknown variant {config.variant!r}. Choose from: {sorted(ABLATION_VARIANTS)}"
        raise ValueError(msg)

    profile = resolve_failure_profile("ablation", config.failure_mode)
    max_steps = effective_max_steps(config.max_steps, profile)
    model_config = config_for_variant(config.variant)
    model = AblationEncoderForSequenceClassification(model_config)
    apply_pathology_model_setup(model, profile)

    dataset = _SequenceClassificationDataset(
        num_samples=config.max_samples,
        seq_len=model_config.seq_len,
        input_dim=model_config.input_dim,
        num_classes=model_config.num_classes,
        seed=config.seed,
    )

    telemetry = config.telemetry or TelemetryConfig.ablation_transformer()
    callbacks, _collector = build_trainer_callbacks(model, telemetry)

    trainer = _AblationTrainer(
        model=model,
        profile=profile,
        param_group_fn=ablation_param_group,
        args=training_arguments(
            output_dir=str(config.output_dir),
            profile=profile,
            max_steps=max_steps,
            seed=config.seed,
            run_name=config.run_name,
            per_device_train_batch_size=32,
            label_names=["labels"],
        ),
        train_dataset=dataset,
        callbacks=callbacks,
    )
    trainer.train()


register_experiment(
    Experiment(
        name="ablation_transformer",
        description=(
            "Mini encoder with toggles for scaled attention, LayerNorm, and residuals. "
            "Use --variant (standard|no_scale|no_norm|no_residual) and --failure-mode."
        ),
        model_family="ablation",
        training_kind="classification",
        profile_family="ablation",
        failure_modes=failure_modes_for_family("ablation"),
        variants=ABLATION_VARIANTS,
        default_telemetry=TelemetryConfig.ablation_transformer,
        run=run_ablation_transformer,
    )
)

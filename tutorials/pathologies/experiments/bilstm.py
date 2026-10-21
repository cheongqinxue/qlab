"""BiLSTM classification — one experiment, all failure modes via CLI."""

from __future__ import annotations

from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer

from pathologies.experiment import Experiment, RunConfig, register_experiment
from pathologies.experiments._common import (
    AG_NEWS_PATH,
    ENCODER_MODEL_ID,
    effective_max_steps,
    failure_modes_for_family,
    resolve_failure_profile,
    training_arguments,
)
from pathologies.experiments.pathology_trainer import PathologyTrainer
from pathologies.instrumentation.config import TelemetryConfig
from pathologies.instrumentation.factory import build_trainer_callbacks
from pathologies.models.bilstm import BiLstmConfig, BiLstmForSequenceClassification


def _tokenize_ag_news(examples: dict, tokenizer, max_length: int) -> dict:
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def _bilstm_shape(failure_mode: str) -> tuple[int, int, int, int]:
    """Return ``hidden_size``, ``num_layers``, ``max_length``, ``batch_size`` for a mode."""
    if failure_mode == "vanish":
        # Deep stack + long sequences so BPTT attenuates gradients toward early layers.
        return 256, 6, 512, 4
    return 128, 2, 128, 16


def run_bilstm(config: RunConfig) -> None:
    """Train BiLSTM on AG News; failure mode from ``config.failure_mode``."""
    profile = resolve_failure_profile("bilstm", config.failure_mode)
    max_steps = effective_max_steps(config.max_steps, profile)
    hidden_size, num_layers, max_length, batch_size = _bilstm_shape(config.failure_mode)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_ID)
    dataset = load_dataset(AG_NEWS_PATH, split="train")
    dataset = dataset.select(range(min(len(dataset), config.max_samples)))
    dataset = dataset.rename_column("label", "labels")

    tokenized = dataset.map(
        partial(_tokenize_ag_news, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )

    model_config = BiLstmConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_labels=4,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    model = BiLstmForSequenceClassification(model_config)

    telemetry = config.telemetry or TelemetryConfig.bilstm()
    callbacks, _collector = build_trainer_callbacks(model, telemetry)

    trainer = PathologyTrainer(
        model=model,
        profile=profile,
        args=training_arguments(
            output_dir=str(config.output_dir),
            profile=profile,
            max_steps=max_steps,
            seed=config.seed,
            run_name=config.run_name,
            per_device_train_batch_size=batch_size,
            label_names=["labels"],
        ),
        train_dataset=tokenized,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()


register_experiment(
    Experiment(
        name="bilstm",
        description=(
            "Custom BiLSTM on AG News. ``vanish`` uses a deep stack and long sequences "
            "so per-layer grad norms show classical depth-wise attenuation."
        ),
        model_family="bilstm",
        training_kind="classification",
        profile_family="bilstm",
        failure_modes=failure_modes_for_family("bilstm"),
        default_telemetry=TelemetryConfig.bilstm,
        run=run_bilstm,
    )
)

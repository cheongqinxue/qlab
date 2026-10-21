"""Tiny BERT encoder MLM — one experiment, all failure modes via CLI."""

from __future__ import annotations

from functools import partial

from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from pathologies.experiment import Experiment, RunConfig, register_experiment
from pathologies.experiments._common import (
    ENCODER_MODEL_ID,
    WIKITEXT_CONFIG,
    WIKITEXT_PATH,
    effective_max_steps,
    failure_modes_for_family,
    resolve_failure_profile,
    training_arguments,
)
from pathologies.experiments.encoder_setup import apply_pathology_model_setup
from pathologies.experiments.pathology_trainer import PathologyTrainer
from pathologies.instrumentation.config import TelemetryConfig
from pathologies.instrumentation.factory import build_trainer_callbacks


def _tokenize_wikitext(examples: dict, tokenizer, block_size: int) -> dict:
    tokenized = tokenizer(examples["text"], truncation=False)
    concatenated: dict[str, list[int]] = {key: sum(tokenized[key], []) for key in tokenized}
    total_length = len(concatenated["input_ids"])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    return {
        key: [values[index : index + block_size] for index in range(0, total_length, block_size)]
        for key, values in concatenated.items()
    }


def run_encoder_mlm(config: RunConfig) -> None:
    """Fine-tune tiny BERT with MLM; failure mode comes from ``config.failure_mode``."""
    profile = resolve_failure_profile("encoder", config.failure_mode)
    max_steps = effective_max_steps(config.max_steps, profile)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(
        ENCODER_MODEL_ID,
        attn_implementation="eager",
    )
    apply_pathology_model_setup(model, profile)

    raw = load_dataset(WIKITEXT_PATH, WIKITEXT_CONFIG, split="train")
    raw = raw.filter(lambda row: len(row["text"].strip()) > 0)
    raw = raw.select(range(min(len(raw), config.max_samples)))

    block_size = 128
    tokenized = raw.map(
        partial(_tokenize_wikitext, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=raw.column_names,
    )

    mlm_probability = profile.mlm_probability if profile.mlm_probability is not None else 0.15
    telemetry = config.telemetry or TelemetryConfig.encoder_mlm()
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
            per_device_train_batch_size=8,
        ),
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        ),
        callbacks=callbacks,
    )
    trainer.train()


register_experiment(
    Experiment(
        name="encoder_mlm",
        description=(
            "Tiny BERT MLM on WikiText. Pass --failure-mode for training pathologies "
            "or encoder best-practice violations."
        ),
        model_family="encoder",
        training_kind="mlm",
        profile_family="encoder",
        failure_modes=failure_modes_for_family("encoder"),
        default_telemetry=TelemetryConfig.encoder_mlm,
        run=run_encoder_mlm,
    )
)

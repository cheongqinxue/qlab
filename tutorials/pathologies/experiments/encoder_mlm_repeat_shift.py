"""Encoder MLM with a fixed repeat-then-shift batch schedule (data lever, same failure modes)."""

from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import SequentialSampler
from transformers import AutoModelForMaskedLM, AutoTokenizer, default_data_collator

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
from pathologies.experiments.checkpoint_callbacks import CheckpointAtStepsCallback
from pathologies.experiments.encoder_setup import apply_pathology_model_setup
from pathologies.experiments.fixed_batch_data import (
    RepeatThenShiftDataset,
    build_repeat_shift_batch_features,
    pick_repeat_shift_source_rows,
)
from pathologies.experiments.pathology_trainer import PathologyTrainer
from pathologies.instrumentation.config import TelemetryConfig
from pathologies.instrumentation.factory import build_trainer_callbacks

BLOCK_SIZE = 128
BATCH_SIZE = 8
# Phase 1: 99 identical batches + 1 shift batch; phase 2: 100 steps on the shift batch.
LEADING_REPEAT_STEPS = 99
TAIL_REPEAT_SHIFT_STEPS = 100
DEFAULT_TOTAL_STEPS = LEADING_REPEAT_STEPS + 1 + TAIL_REPEAT_SHIFT_STEPS
# ``global_step`` checkpoints (0 = before train; 99/100 bracket the shift batch).
SHIFT_CHECKPOINT_STEPS = frozenset({0, 99, 100, 105, 200})


class _SequentialPathologyTrainer(PathologyTrainer):
    """Trainer with deterministic batch order (no shuffled sampler)."""

    def _get_train_sampler(
        self,
        train_dataset: torch.utils.data.Dataset | None = None,
    ) -> torch.utils.data.Sampler[int]:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            msg = "Training requires a train_dataset."
            raise ValueError(msg)
        return SequentialSampler(dataset)


def run_encoder_mlm_repeat_shift(config: RunConfig) -> None:
    """MLM with repeat → shift → repeat on the shift batch.

    Uses the same ``--failure-mode`` presets as ``encoder_mlm``; only the data
    schedule changes (narrow repeat, one shift, then many steps on the new batch).
    """
    profile = resolve_failure_profile("encoder", config.failure_mode)
    max_steps = max(
        effective_max_steps(config.max_steps, profile),
        DEFAULT_TOTAL_STEPS,
    )
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(
        ENCODER_MODEL_ID,
        attn_implementation="eager",
    )
    apply_pathology_model_setup(model, profile)

    raw = load_dataset(WIKITEXT_PATH, WIKITEXT_CONFIG, split="train")
    texts = [row["text"] for row in raw.select(range(min(len(raw), config.max_samples)))]
    texts = [text for text in texts if text.strip()]
    repeat_row, shift_row = pick_repeat_shift_source_rows(
        texts,
        tokenizer,
        BLOCK_SIZE,
        config.seed,
    )
    mlm_probability = profile.mlm_probability if profile.mlm_probability is not None else 0.15
    repeat_batch, shift_batch = build_repeat_shift_batch_features(
        texts,
        tokenizer,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        repeat_row=repeat_row,
        shift_row=shift_row,
        mlm_probability=mlm_probability,
        seed=config.seed,
    )
    train_dataset = RepeatThenShiftDataset(
        repeat_batch,
        shift_batch,
        leading_repeat_steps=LEADING_REPEAT_STEPS,
        tail_repeat_shift_steps=TAIL_REPEAT_SHIFT_STEPS,
    )
    if train_dataset.num_steps != max_steps:
        max_steps = train_dataset.num_steps

    telemetry = config.telemetry or TelemetryConfig.encoder_mlm()
    callbacks, _collector = build_trainer_callbacks(model, telemetry)
    checkpoint_callback = CheckpointAtStepsCallback(
        SHIFT_CHECKPOINT_STEPS,
        tokenizer=tokenizer,
    )
    callbacks = [*callbacks, checkpoint_callback]

    trainer = _SequentialPathologyTrainer(
        model=model,
        profile=profile,
        args=training_arguments(
            output_dir=str(config.output_dir),
            profile=profile,
            max_steps=max_steps,
            seed=config.seed,
            run_name=config.run_name,
            per_device_train_batch_size=BATCH_SIZE,
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )
    checkpoint_callback.bind_trainer(trainer)
    trainer.train()


register_experiment(
    Experiment(
        name="encoder_mlm_repeat_shift",
        description=(
            "Tiny BERT MLM: deterministic data — 99 repeat batches, 1 shift "
            "batch, then 100 steps on the shift batch (200 steps). Same "
            "--failure-mode presets as encoder_mlm."
        ),
        model_family="encoder",
        training_kind="mlm",
        profile_family="encoder",
        failure_modes=failure_modes_for_family("encoder"),
        default_telemetry=TelemetryConfig.encoder_mlm,
        run=run_encoder_mlm_repeat_shift,
    )
)

"""Trainer callbacks for saving checkpoints at explicit global steps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import TrainerCallback
from transformers.trainer import TRAINER_STATE_NAME

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainerState, TrainingArguments


class CheckpointAtStepsCallback(TrainerCallback):
    """Save full training checkpoints at selected ``global_step`` values.

    Each checkpoint includes model weights, tokenizer (if provided), optimizer,
    scheduler, scaler, RNG state, and ``trainer_state.json`` — same artifacts as
    :meth:`~transformers.Trainer._save_checkpoint` (with custom directory names).

    Step ``0`` is written in ``on_train_begin`` (before the first optimizer step).
    Other steps are written in ``on_step_end`` after that step completes.
    """

    def __init__(
        self,
        steps: frozenset[int],
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self._steps = steps
        self._tokenizer = tokenizer
        self._trainer: Trainer | None = None

    def bind_trainer(self, trainer: Trainer) -> None:
        """Attach the trainer (required for optimizer and trainer state saves)."""
        self._trainer = trainer

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control,
        model: PreTrainedModel | None = None,
        **kwargs,
    ) -> None:
        if 0 in self._steps and model is not None:
            self._save(args.output_dir, 0, state)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control,
        model: PreTrainedModel | None = None,
        **kwargs,
    ) -> None:
        if state.global_step not in self._steps or model is None:
            return
        self._save(args.output_dir, state.global_step, state)

    def _save(self, output_dir: str, step: int, state: TrainerState) -> None:
        if self._trainer is None:
            msg = "CheckpointAtStepsCallback.bind_trainer() must be called before training."
            raise RuntimeError(msg)

        checkpoint_dir = Path(output_dir) / f"checkpoint-step-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        out = str(checkpoint_dir)

        self._trainer.save_model(out, _internal_call=True)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(out)

        if not self._trainer.args.save_only_model:
            self._trainer._save_optimizer_and_scheduler(out)
            self._trainer._save_scaler(out)
            self._trainer._save_rng_state(out)

        if self._trainer.args.should_save:
            state.save_to_json(os.path.join(out, TRAINER_STATE_NAME))

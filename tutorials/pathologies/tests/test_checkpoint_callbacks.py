"""Tests for step-indexed checkpoint callback."""

from pathlib import Path

import pytest
from transformers import TrainerCallback

from pathologies.experiments.checkpoint_callbacks import CheckpointAtStepsCallback


def test_checkpoint_callback_is_trainer_callback() -> None:
    """Callback should plug into HuggingFace Trainer."""
    callback = CheckpointAtStepsCallback(frozenset({0, 99}))
    assert isinstance(callback, TrainerCallback)


def test_checkpoint_requires_bound_trainer(tmp_path: Path) -> None:
    """Full checkpoints need a bound trainer."""
    callback = CheckpointAtStepsCallback(frozenset({0}))
    state = type("State", (), {"global_step": 0, "save_to_json": lambda self, p: None})()
    with pytest.raises(RuntimeError, match="bind_trainer"):
        callback._save(str(tmp_path), 0, state)

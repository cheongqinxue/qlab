"""Ensure custom telemetry reaches Trainer log history / W&B integration."""

from __future__ import annotations

import os

import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import CallbackHandler

from pathologies.experiments._common import BASELINE
from pathologies.experiments.pathology_trainer import (
    PathologySFTTrainer,
    PathologyTrainer,
    alias_trl_entropy_as_logit_entropy,
)
from pathologies.instrumentation.callbacks import GradientTelemetryCallback
from pathologies.instrumentation.config import TelemetryConfig


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_custom_metrics_in_log_history(tmp_path, monkeypatch) -> None:
    """PathologyTrainer should enrich logs before the W&B integration runs."""
    monkeypatch.setenv("WANDB_MODE", "disabled")

    model = _TinyModel()
    inputs = torch.randn(2, 4)
    targets = torch.tensor([0, 1])
    dataset = [{"x": inputs[i], "y": targets[i]} for i in range(2)]

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(dataset)

        def __getitem__(self, index: int) -> dict:
            return dataset[index]

    def collate(batch: list[dict]) -> dict:
        return {
            "x": torch.stack([row["x"] for row in batch]),
            "y": torch.stack([row["y"] for row in batch]),
        }

    config = TelemetryConfig(
        log_layer_grad_norms=True,
        log_update_ratio=True,
    )

    class _Trainer(PathologyTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            del kwargs
            logits = model(inputs["x"])
            loss = nn.functional.cross_entropy(logits, inputs["y"])
            return (loss, logits) if return_outputs else loss

    trainer = _Trainer(
        model=model,
        profile=BASELINE,
        args=TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=2,
            per_device_train_batch_size=2,
            logging_steps=1,
            report_to=["wandb"] if os.environ.get("WANDB_MODE") != "disabled" else [],
            save_strategy="no",
            remove_unused_columns=False,
            use_cpu=True,
        ),
        train_dataset=_Dataset(),
        data_collator=collate,
        callbacks=[GradientTelemetryCallback(config)],
    )
    trainer.train()

    logged = [
        entry
        for entry in trainer.state.log_history
        if "update_to_weight_ratio" in entry
    ]
    assert logged, "expected update_to_weight_ratio in log_history"


def test_sft_trainer_enriches_before_wandb(monkeypatch) -> None:
    """PathologySFTTrainer should enrich logs before Trainer.log forwards to W&B."""
    logs_sent: dict[str, float] = {}

    def capture_trainer_log(self, logs, start_time=None) -> None:
        del self, start_time
        logs_sent.update(logs)

    monkeypatch.setattr(Trainer, "log", capture_trainer_log)

    model = _TinyModel()
    callback = GradientTelemetryCallback(TelemetryConfig.minimal())
    trainer = PathologySFTTrainer.__new__(PathologySFTTrainer)
    trainer.model = model
    trainer._metrics = {"train": {}, "eval": {}}
    trainer.callback_handler = CallbackHandler([callback], model, None, None, None)
    callback._param_snapshot = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    callback._grad_norm = 1.0
    callback._grad_max = 0.5
    callback._grad_max_name = "linear.weight"

    trainer.log({"loss": 0.5})

    assert "update_to_weight_ratio" in logs_sent
    assert "grad_norm" in logs_sent


def test_sft_logit_entropy_aliases_trl_entropy(monkeypatch) -> None:
    """SFT should expose TRL vocab entropy as logit_entropy for pathology panels."""
    logs_sent: dict[str, float] = {}

    def capture_trainer_log(self, logs, start_time=None) -> None:
        del self, start_time
        logs_sent.update(logs)

    monkeypatch.setattr(Trainer, "log", capture_trainer_log)

    trainer = PathologySFTTrainer.__new__(PathologySFTTrainer)
    trainer.model = _TinyModel()
    trainer._metrics = {"train": {"entropy": [1.25]}, "eval": {}}
    trainer.callback_handler = CallbackHandler([], trainer.model, None, None, None)

    trainer.log({"loss": 0.5})

    assert logs_sent["entropy"] == 1.25
    assert logs_sent["logit_entropy"] == 1.25


def test_alias_trl_entropy_skips_when_hook_metric_present() -> None:
    """Hook-derived logit_entropy must not be overwritten by the TRL alias."""
    metrics = {"entropy": 1.0, "logit_entropy": 2.0}
    alias_trl_entropy_as_logit_entropy(metrics)
    assert metrics["logit_entropy"] == 2.0

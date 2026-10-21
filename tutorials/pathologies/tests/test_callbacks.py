"""Tests for Trainer telemetry callbacks."""

import torch
from torch import nn
from transformers import Trainer, TrainingArguments

from pathologies.instrumentation.callbacks import GradientTelemetryCallback
from pathologies.instrumentation.config import TelemetryConfig


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestGradientTelemetryCallback:
    """Gradient norms should be attached to Trainer logs."""

    def test_logs_grad_norm(self, tmp_path) -> None:
        """After one training step, logs should include grad_norm."""
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

        class _Trainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                del kwargs
                logits = model(inputs["x"])
                loss = nn.functional.cross_entropy(logits, inputs["y"])
                return (loss, logits) if return_outputs else loss

        callback = GradientTelemetryCallback(TelemetryConfig.minimal())
        args = TrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=2,
            logging_steps=1,
            report_to=[],
            save_strategy="no",
            remove_unused_columns=False,
        )
        trainer = _Trainer(
            model=model,
            args=args,
            train_dataset=_Dataset(),
            data_collator=collate,
            callbacks=[callback],
        )
        trainer.train()

        assert any("grad_norm" in entry for entry in trainer.state.log_history)

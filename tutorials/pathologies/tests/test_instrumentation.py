"""Tests for pluggable instrumentation."""

import torch
from torch import nn

from pathologies.instrumentation import TelemetryConfig, build_trainer_callbacks
from pathologies.instrumentation.patterns import list_matching_modules, name_matches


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.encoder(x)))


class TestInstrumentation:
    """Attach/detach telemetry building blocks."""

    def test_name_matches_regex(self) -> None:
        """Regex patterns should use full-match semantics."""
        pattern = r"re:blocks\.[0-9]+\.attn"
        assert name_matches("blocks.0.attn", [pattern])
        assert not name_matches("blocks.0.attn.extra", [pattern])

    def test_build_callbacks_minimal(self) -> None:
        """Minimal config should yield a gradient callback only."""
        model = _TinyModel()
        callbacks, collector = build_trainer_callbacks(model, TelemetryConfig.minimal())
        assert len(callbacks) == 1
        assert collector is None

    def test_llm_preset_hooks_lm_head(self) -> None:
        """LLM preset should hook top-level or nested ``lm_head`` (SmolLM vs some Llama layouts)."""

        class _NestedLmHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Module()
                self.model.lm_head = nn.Linear(4, 8)

        class _TopLevelLmHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Module()
                self.lm_head = nn.Linear(4, 8)

        config = TelemetryConfig.llm()
        assert list_matching_modules(_NestedLmHead(), [config.logits_module]) == ["model.lm_head"]
        assert list_matching_modules(_TopLevelLmHead(), [config.logits_module]) == ["lm_head"]

    def test_build_callbacks_with_hooks(self, tmp_path) -> None:
        """Full ablation preset should register forward hooks."""
        model = _TinyModel()
        config = TelemetryConfig(
            log_activation_rms=True,
            activation_modules=["encoder", "head"],
        )
        callbacks, collector = build_trainer_callbacks(model, config)
        assert collector is not None
        assert len(callbacks) == 2

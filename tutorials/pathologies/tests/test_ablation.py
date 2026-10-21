"""Tests for ablation encoder and variant configs."""

import torch

from pathologies.models.ablation_encoder import (
    AblationEncoderForSequenceClassification,
    config_for_variant,
)


class TestAblationEncoder:
    """Ablation encoder forward and variant switches."""

    def test_standard_forward(self) -> None:
        """Standard variant should produce logits and loss."""
        config = config_for_variant("standard")
        model = AblationEncoderForSequenceClassification(config)
        inputs = torch.randn(2, config.seq_len, config.input_dim)
        labels = torch.tensor([0, 1])
        output = model(inputs=inputs, labels=labels)
        assert output.loss is not None
        assert output.logits.shape == (2, config.num_classes)

    def test_no_scale_variant_disables_scaling(self) -> None:
        """no_scale variant should set scaled_attention=False."""
        config = config_for_variant("no_scale")
        assert config.scaled_attention is False

    def test_no_norm_variant(self) -> None:
        """no_norm variant should disable LayerNorm."""
        config = config_for_variant("no_norm")
        assert config.layer_norm is False

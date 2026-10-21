"""Tests for the custom BiLSTM PreTrainedModel."""

import torch

from pathologies.models.bilstm import BiLstmConfig, BiLstmForSequenceClassification


class TestBiLstmModel:
    """BiLSTM should behave like a HuggingFace sequence classifier."""

    def test_forward_with_labels_returns_loss(self) -> None:
        """Forward pass with labels should produce a scalar loss."""
        config = BiLstmConfig(vocab_size=100, hidden_size=32, num_labels=4)
        model = BiLstmForSequenceClassification(config)
        input_ids = torch.randint(0, 100, (2, 12))
        attention_mask = torch.ones(2, 12, dtype=torch.long)
        labels = torch.tensor([0, 2])

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        assert output.loss is not None
        assert output.logits.shape == (2, 4)

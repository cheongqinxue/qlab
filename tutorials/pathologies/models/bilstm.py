"""BiLSTM sequence classifier exposed as a HuggingFace ``PreTrainedModel``."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BiLstmConfig(PretrainedConfig):
    """Configuration for :class:`BiLstmForSequenceClassification`."""

    model_type = "bilstm_pathologies"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        num_labels: int = 4,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        **kwargs,
    ) -> None:
        """Initialize BiLSTM hyperparameters.

        Args:
            vocab_size: Embedding vocabulary size (match the tokenizer).
            hidden_size: LSTM hidden size per direction; classifier input is ``2 * hidden_size``.
            num_hidden_layers: Number of stacked BiLSTM layers.
            num_labels: Number of classification labels.
            dropout: Dropout applied before the classifier head.
            pad_token_id: Padding token id for the embedding layer.
            **kwargs: Forwarded to :class:`~transformers.PretrainedConfig`.
        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_labels = num_labels
        self.dropout = dropout


class BiLstmForSequenceClassification(PreTrainedModel):
    """Bidirectional LSTM classifier compatible with ``transformers.Trainer``."""

    config_class = BiLstmConfig

    def __init__(self, config: BiLstmConfig) -> None:
        """Build embedding, BiLSTM stack, and linear classifier."""
        super().__init__(config)
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=config.num_hidden_layers,
            dropout=config.dropout if config.num_hidden_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Run the BiLSTM classifier.

        Args:
            input_ids: Token ids ``(batch, seq)``.
            attention_mask: Mask with ``1`` for real tokens and ``0`` for padding.
            labels: Optional class ids ``(batch,)``.
            **kwargs: Ignored; kept for Trainer compatibility.

        Returns:
            :class:`~transformers.modeling_outputs.SequenceClassifierOutput` with
            ``loss`` and ``logits``.
        """
        del kwargs
        if input_ids is None:
            msg = "input_ids are required"
            raise ValueError(msg)

        embedded = self.embedding(input_ids)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)

        forward_last = hidden[-2]
        backward_last = hidden[-1]
        pooled = torch.cat([forward_last, backward_last], dim=-1)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

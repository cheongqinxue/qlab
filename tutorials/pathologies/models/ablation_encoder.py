"""Configurable mini-encoder for transformer component ablation studies."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class AblationEncoderConfig(PretrainedConfig):
    """Hyperparameters and ablation switches for :class:`AblationEncoderForSequenceClassification`."""

    model_type = "ablation_encoder"

    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 5,
        seq_len: int = 64,
        dropout: float = 0.1,
        scaled_attention: bool = True,
        layer_norm: bool = True,
        residual: bool = True,
        **kwargs,
    ) -> None:
        """Initialize ablation encoder settings.

        Args:
            input_dim: Per-timestep input feature size.
            d_model: Model hidden size (must divide ``num_heads``).
            num_heads: Number of attention heads.
            ff_dim: Feed-forward inner dimension.
            num_layers: Number of transformer blocks.
            num_classes: Classification labels.
            seq_len: Sequence length (for synthetic data).
            dropout: Dropout probability.
            scaled_attention: When False, omit ``1/sqrt(d_head)`` scaling (ablation).
            layer_norm: When False, skip LayerNorm (ablation).
            residual: When False, skip residual connections (ablation).
            **kwargs: Forwarded to :class:`~transformers.PretrainedConfig`.
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.dropout = dropout
        self.scaled_attention = scaled_attention
        self.layer_norm = layer_norm
        self.residual = residual


class AblationSelfAttention(nn.Module):
    """Multi-head self-attention with optional scaled dot-product."""

    def __init__(self, config: AblationEncoderConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            msg = "d_model must be divisible by num_heads"
            raise ValueError(msg)
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.scaled_attention = config.scaled_attention
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention and return output plus attention weights."""
        batch, seq_len, _ = hidden.shape
        qkv = self.qkv(hidden)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(query, key.transpose(-2, -1))
        if self.scaled_attention:
            scores = scores / math.sqrt(self.head_dim)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).reshape(batch, seq_len, -1)
        output = self.out_proj(context)
        # Average over heads for telemetry (batch, seq, seq)
        avg_weights = attn_weights.mean(dim=1)
        return output, avg_weights


class AblationEncoderBlock(nn.Module):
    """Pre-norm transformer block with ablation toggles."""

    def __init__(self, config: AblationEncoderConfig) -> None:
        super().__init__()
        self.residual_enabled = config.residual
        self.attn = AblationSelfAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.d_model),
        )
        self.norm1 = nn.LayerNorm(config.d_model) if config.layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(config.d_model) if config.layer_norm else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run attention and feed-forward sublayers."""
        attn_out, attn_weights = self.attn(hidden)
        if self.residual_enabled:
            hidden = self.norm1(hidden + self.dropout(attn_out))
            ff_out = self.ff(hidden)
            hidden = self.norm2(hidden + self.dropout(ff_out))
        else:
            hidden = self.norm1(self.dropout(attn_out))
            hidden = self.norm2(self.dropout(self.ff(hidden)))
        return hidden, attn_weights


class AblationEncoderForSequenceClassification(PreTrainedModel):
    """Mini encoder with toggles demonstrating why standard transformer pieces matter."""

    config_class = AblationEncoderConfig

    def __init__(self, config: AblationEncoderConfig) -> None:
        super().__init__(config)
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.blocks = nn.ModuleList(
            AblationEncoderBlock(config) for _ in range(config.num_layers)
        )
        self.head = nn.Linear(config.d_model, config.num_classes)
        self.post_init()

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Classify a batch of continuous input sequences.

        Args:
            inputs: Float tensor ``(batch, seq, input_dim)``.
            labels: Optional class ids ``(batch,)``.
            **kwargs: Ignored; kept for Trainer compatibility.

        Returns:
            :class:`~transformers.modeling_outputs.SequenceClassifierOutput`.
        """
        del kwargs
        if inputs is None:
            msg = "inputs tensor is required"
            raise ValueError(msg)

        hidden = self.input_proj(inputs)
        for block in self.blocks:
            hidden, _attn = block(hidden)
        logits = self.head(hidden.mean(dim=1))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def config_for_variant(variant: str) -> AblationEncoderConfig:
    """Return an :class:`AblationEncoderConfig` for a named ablation variant.

    Args:
        variant: One of ``standard``, ``no_scale``, ``no_norm``, ``no_residual``.

    Returns:
        Config with the corresponding ablation switches.

    Raises:
        ValueError: If ``variant`` is unknown.
    """
    variants: dict[str, dict[str, bool | str]] = {
        "standard": {
            "scaled_attention": True,
            "layer_norm": True,
            "residual": True,
        },
        "no_scale": {
            "scaled_attention": False,
            "layer_norm": True,
            "residual": True,
        },
        "no_norm": {
            "scaled_attention": True,
            "layer_norm": False,
            "residual": True,
        },
        "no_residual": {
            "scaled_attention": True,
            "layer_norm": True,
            "residual": False,
        },
    }
    if variant not in variants:
        msg = f"Unknown ablation variant {variant!r}. Choose from: {sorted(variants)}"
        raise ValueError(msg)
    return AblationEncoderConfig(**variants[variant])

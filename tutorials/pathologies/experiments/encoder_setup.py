"""Model-side hooks for encoder pathology induction."""

from __future__ import annotations

import torch
from torch import nn

from pathologies.experiments._common import FailureProfile


def apply_pathology_model_setup(model: nn.Module, profile: FailureProfile) -> None:
    """Apply init or freezing required by an encoder failure profile.

    Args:
        model: BERT MLM model (or compatible encoder).
        profile: Selected failure profile.
    """
    if profile.trainable_prefixes:
        _restrict_trainable_params(model, profile.trainable_prefixes)
    if profile.attention_sink_init:
        apply_attention_sink_init(model)


def apply_attention_sink_init(model: nn.Module) -> None:
    """Bias the model toward attending heavily to the first sequence position."""
    with torch.no_grad():
        embeddings = getattr(model, "bert", None)
        if embeddings is None:
            return
        embeddings = embeddings.embeddings
        position = embeddings.position_embeddings.weight
        position[0].mul_(8.0)
        position[1:].mul_(0.25)
        word = embeddings.word_embeddings.weight
        word[0].mul_(3.0)


def _restrict_trainable_params(model: nn.Module, prefixes: tuple[str, ...]) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)

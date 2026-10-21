"""Tests for per-depth gradient norm diagnostics."""

import torch
from torch import nn

from pathologies.instrumentation import diagnostics
from pathologies.models.bilstm import BiLstmConfig, BiLstmForSequenceClassification


def test_llama_depth_grad_norms() -> None:
    """Llama-style blocks should produce per-layer grad norm keys."""
    import torch
    from torch import nn

    from pathologies.instrumentation.diagnostics import depth_grad_norms

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

    class _LlamaLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)
            self.layers = nn.ModuleList([_Block(), _Block()])
            self.lm_head = nn.Linear(4, 8)

    model = _LlamaLike()
    for param in model.parameters():
        param.grad = torch.randn_like(param)

    metrics = depth_grad_norms(model)
    assert "grad_norm/layer.0" in metrics
    assert "grad_norm/layer.1" in metrics
    assert "grad_norm/head" in metrics


def test_bilstm_depth_grad_early_to_late_ratio() -> None:
    """Shallow synthetic grads should yield a measurable early/late layer ratio."""
    config = BiLstmConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_labels=2)
    model = BiLstmForSequenceClassification(config)
    for name, param in model.named_parameters():
        param.grad = torch.zeros_like(param)
        match = __import__("re").search(r"lstm\.weight_(?:ih|hh)_l(\d+)", name)
        if match:
            layer = int(match.group(1))
            param.grad.fill_(0.01 * (layer + 1))

    depth_norms = diagnostics.depth_grad_norms(model)
    assert "grad_norm/layer.0" in depth_norms
    assert "grad_norm/layer.2" in depth_norms
    ratio = diagnostics.depth_grad_early_to_late_ratio(depth_norms)
    assert ratio is not None
    assert ratio < 1.0

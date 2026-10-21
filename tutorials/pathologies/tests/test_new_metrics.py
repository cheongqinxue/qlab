"""Tests for attention-head, Adam state, and update-rank diagnostics."""

from __future__ import annotations

import torch
from torch import nn

from pathologies.instrumentation import diagnostics


class TestAttentionPerHeadMetrics:
    """Per-head attention imbalance."""

    def test_imbalance_high_when_one_head_peaks(self) -> None:
        """One peaked head should raise max/min imbalance."""
        batch, heads, seq = 2, 4, 3
        attention = torch.full((batch, heads, seq, seq), 1.0 / seq)
        attention[:, 0, :, :] = 0.0
        attention[:, 0, :, 0] = 1.0
        metrics = diagnostics.attention_per_head_metrics(attention)
        assert metrics["_imbalance"] > 2.0
        assert len([key for key in metrics if key.startswith("head.")]) == heads

    def test_returns_empty_for_averaged_attention(self) -> None:
        """Three-dimensional tensors skip per-head breakdown."""
        attention = torch.full((2, 3, 3), 1.0 / 3)
        assert diagnostics.attention_per_head_metrics(attention) == {}


class _FakeBertLayers(nn.Module):
    """Minimal module whose parameter names mimic BERT layer paths."""

    def __init__(self) -> None:
        super().__init__()
        self.layer0 = nn.Linear(4, 4)
        self.layer1 = nn.Linear(4, 4)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        del prefix, recurse
        yield "bert.encoder.layer.0.dense.weight", self.layer0.weight
        yield "bert.encoder.layer.1.dense.weight", self.layer1.weight


class TestAdamSecondMoment:
    """Adam exp_avg_sq aggregation."""

    def test_groups_llama_layers(self) -> None:
        """SmolLM-style ``model.layers.N`` params should aggregate by block index."""

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)

        class _LlamaLike(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([_Block(), _Block()])

        model = _LlamaLike()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = sum(layer.linear(torch.ones(2, 4)).sum() for layer in model.layers)
        loss.backward()
        optimizer.step()

        metrics = diagnostics.adam_second_moment_by_layer(model, optimizer)
        assert "adam_second_moment/layer.0" in metrics
        assert "adam_second_moment/layer.1" in metrics

    def test_groups_bert_layers(self) -> None:
        """Second moments should aggregate by encoder layer index."""
        model = _FakeBertLayers()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model.layer0(torch.ones(2, 4)).sum() + model.layer1(torch.ones(2, 4)).sum()
        loss.backward()
        optimizer.step()

        metrics = diagnostics.adam_second_moment_by_layer(model, optimizer)
        assert "adam_second_moment/layer.0" in metrics
        assert "adam_second_moment/layer.1" in metrics
        assert metrics["adam_second_moment/layer.0"] > 0.0


class TestAdamHealthMetrics:
    """Adam fragility and effective-step diagnostics."""

    def test_post_step_r_increases_with_tiny_second_moment(self) -> None:
        """Tiny ``exp_avg_sq`` with moderate ``exp_avg`` should yield large ``adam_max_r``."""
        model = nn.Linear(4, 2, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-8)
        loss = model(torch.ones(2, 4)).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            param = next(model.parameters())
            state = optimizer.state[param]
            state["exp_avg_sq"].fill_(1e-12)
            state["exp_avg"].fill_(1e-3)
            state["step"] = torch.tensor(10)
        metrics = diagnostics.adam_post_step_health_metrics(model, optimizer)
        assert metrics["adam_max_r"] > 1.0
        assert 0.0 <= metrics["adam_low_moment_frac"] <= 1.0

    def test_pre_step_g2_over_v_uses_gradients(self) -> None:
        """Gradient spike metric should be positive when grad is non-zero."""
        model = nn.Linear(4, 2, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-8)
        loss = (model(torch.ones(2, 4)) ** 2).sum()
        loss.backward()
        metrics = diagnostics.adam_pre_step_spike_metrics(model, optimizer)
        assert metrics["adam_max_g2_over_v"] > 0.0


class TestUpdateRank:
    """Effective rank of weight updates."""

    def test_rank_of_structured_update(self) -> None:
        """A rank-1 delta should have low participation ratio."""
        before = {"weight": torch.zeros(8, 8)}
        model = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.zeros(8, 8))
            model.weight[0, :] = 1.0
        ranks = diagnostics.update_ranks(before, model, ["weight"])
        assert ranks["update_rank/weight"] <= 2.0

    def test_zero_update_has_zero_rank(self) -> None:
        """Unchanged weights produce zero update rank."""
        weight = torch.ones(4, 4)
        before = {"weight": weight.clone()}
        model = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            model.weight.copy_(weight)
        ranks = diagnostics.update_ranks(before, model, ["weight"])
        assert ranks["update_rank/weight"] == 0.0

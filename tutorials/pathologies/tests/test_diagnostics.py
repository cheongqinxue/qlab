"""Unit tests for diagnostic helpers."""

import torch

from pathologies.diagnostics import global_grad_norm, max_param_grad_norm


class TestDiagnostics:
    """Gradient diagnostic helpers."""

    def test_global_grad_norm(self) -> None:
        """Global norm should match manual L2 over gradients."""
        param = torch.nn.Parameter(torch.ones(2, 2))
        param.grad = torch.full((2, 2), 3.0)
        assert global_grad_norm([param]) == 6.0

    def test_max_param_grad_norm(self) -> None:
        """Max norm should report the largest parameter gradient."""
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False),
            torch.nn.Linear(2, 2, bias=False),
        )
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        model[1].weight.grad.fill_(5.0)

        max_norm, name = max_param_grad_norm(model)
        assert max_norm == 10.0
        assert name == "1.weight"

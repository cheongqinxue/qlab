"""Forward hooks that stash module outputs for telemetry callbacks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pathologies.instrumentation.patterns import name_matches


class ForwardHookCollector:
    """Stores forward outputs keyed by diagnostic category."""

    def __init__(self) -> None:
        self.outputs: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def clear(self) -> None:
        """Drop captured tensors from the last forward pass."""
        self.outputs.clear()

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.outputs.clear()

    def register(
        self,
        model: nn.Module,
        *,
        activation_modules: list[str],
        attention_modules: list[str],
        logits_module: str | None,
    ) -> None:
        """Register forward hooks on modules matched by pattern lists."""
        for name, module in model.named_modules():
            if not name:
                continue
            if activation_modules and name_matches(name, activation_modules):
                self._handles.append(
                    module.register_forward_hook(self._activation_hook(name))
                )
            if attention_modules and name_matches(name, attention_modules):
                self._handles.append(
                    module.register_forward_hook(self._attention_hook(name))
                )
            if logits_module and name_matches(name, [logits_module]):
                self._handles.append(
                    module.register_forward_hook(self._logits_hook(name))
                )

    def _activation_hook(self, module_name: str):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = _extract_tensor(output)
            if tensor is not None:
                self.outputs[f"activation:{module_name}"] = tensor

        return hook

    def _logits_hook(self, module_name: str):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = _extract_tensor(output)
            if tensor is not None:
                self.outputs[f"logits:{module_name}"] = tensor

        return hook

    def _attention_hook(self, module_name: str):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = _extract_attention_tensor(output)
            if tensor is not None:
                self.outputs[f"attention:{module_name}"] = tensor

        return hook


def _extract_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output.detach()
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0].detach()
    return None


def _extract_attention_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, tuple) and len(output) >= 2:
        weights = output[1]
        if isinstance(weights, torch.Tensor) and weights.ndim >= 2:
            return weights.detach()
    if isinstance(output, torch.Tensor) and output.ndim >= 2:
        return output.detach()
    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim >= 3:
                return item.detach()
    return None

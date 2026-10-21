"""Backward-compatible re-exports — prefer ``pathologies.instrumentation.diagnostics``."""

from pathologies.instrumentation import diagnostics as _diagnostics

global_grad_norm = _diagnostics.global_grad_norm
max_param_grad_norm = _diagnostics.max_param_grad_norm
mean_update_to_weight_ratio = _diagnostics.mean_update_to_weight_ratio

__all__ = ["global_grad_norm", "max_param_grad_norm", "mean_update_to_weight_ratio"]

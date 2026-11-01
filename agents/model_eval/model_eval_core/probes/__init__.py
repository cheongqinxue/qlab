"""Case probes and infrastructure failure tagging."""

from model_eval_core.probes.case import (
    FAILURE_MODE_OLLAMA_TOOL_BINDING_UNSUPPORTED,
    ProbeResult,
    ToolBindingSkippedError,
    is_ollama_tool_binding_unsupported,
    probe_case,
    probe_tool_binding_unsupported,
)
from model_eval_core.probes.infrastructure import (
    FAILURE_MODE_REPETITION_LOOP,
    FAILURE_MODE_THINKING_BUDGET,
    append_infrastructure_probes,
    detect_repetition_loop,
    infrastructure_counts_from_rows,
    max_completion_tokens,
)

__all__ = [
    "FAILURE_MODE_OLLAMA_TOOL_BINDING_UNSUPPORTED",
    "FAILURE_MODE_REPETITION_LOOP",
    "FAILURE_MODE_THINKING_BUDGET",
    "ProbeResult",
    "ToolBindingSkippedError",
    "append_infrastructure_probes",
    "detect_repetition_loop",
    "infrastructure_counts_from_rows",
    "is_ollama_tool_binding_unsupported",
    "max_completion_tokens",
    "probe_case",
    "probe_tool_binding_unsupported",
]

"""Detect Ollama OpenAI-compat errors when tool/grammar binding is unsupported."""

from __future__ import annotations

import re
from typing import Any

from model_eval_core.probes import ProbeResult

FAILURE_MODE_OLLAMA_TOOL_BINDING_UNSUPPORTED = "ollama_tool_binding_unsupported"


class ToolBindingSkippedError(Exception):
    """Raised internally when the suite skips ReAct after a prior tool-binding refusal."""


# Observed when Ollama cannot initialize tool grammars for a model (400 from /v1/chat/completions).
_TOOL_BINDING_MESSAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"does not support tools", re.IGNORECASE),
    re.compile(r"failed to parse grammar", re.IGNORECASE),
)


def _status_code(exc: BaseException) -> int | None:
    """Return HTTP status when present on OpenAI SDK or wrapper exceptions."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    if response is not None:
        resp_code = getattr(response, "status_code", None)
        if isinstance(resp_code, int):
            return resp_code
    return None


def _collect_error_text(exc: BaseException) -> str:
    """Merge exception string and nested API error bodies for pattern matching."""
    chunks: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chunks.append(str(current))
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                if msg:
                    chunks.append(str(msg))
            elif err:
                chunks.append(str(err))
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return " ".join(chunks)


def is_ollama_tool_binding_unsupported(exc: BaseException) -> bool:
    """Return True when Ollama rejected tool binding for this model (not other 400s).

    Args:
        exc: Exception raised from ``graph.ainvoke`` during ReAct tool_use runs.

    Returns:
        True only for documented Ollama tool/grammar refusal messages on HTTP 400.
    """
    status = _status_code(exc)
    if status is not None and status != 400:
        return False
    text = _collect_error_text(exc)
    return any(pattern.search(text) for pattern in _TOOL_BINDING_MESSAGE_PATTERNS)


def _safe_error_signal(exc: BaseException) -> dict[str, Any]:
    """Build non-secret diagnostic fields for results JSONL."""
    return {
        "error_type": type(exc).__name__,
        "error_message": _collect_error_text(exc)[:500],
    }


def probe_tool_binding_unsupported(
    exc: BaseException,
    *,
    skipped_without_invoke: bool = False,
) -> ProbeResult:
    """Build probe output for models that cannot run Ollama tool binding.

    Args:
        exc: Original API error, or a stand-in when skipping after the first hit.
        skipped_without_invoke: True when the suite skipped ReAct after an earlier case.

    Returns:
        ProbeResult tagged with ``ollama_tool_binding_unsupported``.
    """
    signals: dict[str, Any] = {
        "tools_used": [],
        "answer_chars": 0,
        "tool_call_count": 0,
        **_safe_error_signal(exc),
    }
    if skipped_without_invoke:
        signals["skipped_after_first_tool_binding_error"] = True
    note = FAILURE_MODE_OLLAMA_TOOL_BINDING_UNSUPPORTED
    if skipped_without_invoke:
        note = f"{note}; skipped_after_first_tool_binding_error"
    return ProbeResult(
        failure_modes=[FAILURE_MODE_OLLAMA_TOOL_BINDING_UNSUPPORTED],
        signals=signals,
        notes=note,
    )

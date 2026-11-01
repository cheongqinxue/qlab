"""Infrastructure failure modes derived from stored traces (messages + final answer).

These tags describe *how* a run failed (tool stack, thinking budget, repetition),
complementing content rubrics. Safe to re-run on ``responses.jsonl`` without re-invoking models.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage

if TYPE_CHECKING:
    from model_eval_core.probes import ProbeResult


def _strip_thinking(text: str) -> str:
    """Return user-visible text after optional chain-of-thought blocks."""
    think_close = "</" + "redacted_thinking" + ">"
    legacy_think_close = "</" + "think" + ">"
    for close_tag in (legacy_think_close, think_close):
        idx = text.lower().find(close_tag.lower())
        if idx != -1:
            return text[idx + len(close_tag) :].strip()
    return text.strip()


FAILURE_MODE_THINKING_BUDGET = "thinking_budget_exhausted"
FAILURE_MODE_REPETITION_LOOP = "repetition_loop"

_DEFAULT_COMPLETION_TOKEN_FLOOR = 30_000
_DEFAULT_MIN_COMPLETION_FOR_EMPTY_VISIBLE = 4_000


def thinking_token_threshold() -> int:
    """Max-completion-tokens floor for ``thinking_budget_exhausted``."""
    raw = os.environ.get("MODEL_EVAL_THINKING_TOKEN_THRESHOLD", "")
    if raw.isdigit():
        return int(raw)
    return _DEFAULT_COMPLETION_TOKEN_FLOOR


def max_completion_tokens(messages: Sequence[BaseMessage | Mapping[str, Any]] | None) -> int:
    """Largest ``completion_tokens`` seen on any AI message in a trace."""
    if not messages:
        return 0
    best = 0
    for msg in messages:
        usage: dict[str, Any] | None = None
        if isinstance(msg, AIMessage):
            meta = getattr(msg, "response_metadata", None) or {}
            usage = meta.get("token_usage") if isinstance(meta, dict) else None
        elif isinstance(msg, Mapping):
            if msg.get("type") != "ai":
                continue
            usage = msg.get("usage")
            if usage is None:
                additional = msg.get("additional_kwargs")
                if isinstance(additional, dict):
                    usage = additional.get("usage")
        if not isinstance(usage, dict):
            continue
        ct = usage.get("completion_tokens")
        if isinstance(ct, (int, float)):
            best = max(best, int(ct))
    return best


def _ifeval_intentional_repeat(case_id: str, metadata: Mapping[str, Any] | None) -> bool:
    if case_id.startswith("ifeval-") and "repeat" in case_id.lower():
        return True
    if metadata:
        ids = metadata.get("instruction_id_list") or []
        if "combination:repeat_prompt" in ids:
            return True
    return False


def detect_repetition_loop(
    visible_answer: str,
    *,
    case_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> bool:
    """Heuristic mode collapse: repeated lines or repeated medium-length substrings."""
    if _ifeval_intentional_repeat(case_id, metadata):
        return False
    text = visible_answer.strip()
    if len(text) < 150:
        return False
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 15]
    if lines:
        from collections import Counter

        _line, count = Counter(lines).most_common(1)[0]
        if count >= 4:
            return True
    for width in (50, 35, 25):
        limit = min(800, max(0, len(text) - width))
        for start in range(limit):
            chunk = text[start : start + width]
            if len(set(chunk)) < 8:
                continue
            if text.count(chunk) >= 3:
                return True
    return False


def detect_thinking_budget_exhausted(
    final_answer_raw: str,
    messages: Sequence[BaseMessage | Mapping[str, Any]] | None,
) -> tuple[bool, int, int]:
    """Return (detected, max_completion_tokens, visible_answer_chars)."""
    visible = _strip_thinking(final_answer_raw or "")
    visible_chars = len(visible.strip())
    max_ct = max_completion_tokens(messages)
    threshold = thinking_token_threshold()
    if max_ct >= threshold and visible_chars < 20:
        return True, max_ct, visible_chars
    if max_ct >= _DEFAULT_MIN_COMPLETION_FOR_EMPTY_VISIBLE and visible_chars == 0:
        return True, max_ct, visible_chars
    if final_answer_raw.strip() and visible_chars == 0 and max_ct >= 2_000:
        return True, max_ct, visible_chars
    return False, max_ct, visible_chars


def append_infrastructure_probes(
    result: ProbeResult,
    *,
    case_id: str,
    final_answer_raw: str,
    messages: Sequence[BaseMessage | Mapping[str, Any]] | None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Add infrastructure failure modes and ``signals['infrastructure']`` to a probe result."""
    infra: dict[str, Any] = dict(result.signals.get("infrastructure") or {})

    exhausted, max_ct, visible_chars = detect_thinking_budget_exhausted(
        final_answer_raw,
        messages,
    )
    infra["max_completion_tokens"] = max_ct
    infra["visible_answer_chars"] = visible_chars
    if exhausted and FAILURE_MODE_THINKING_BUDGET not in result.failure_modes:
        result.failure_modes.append(FAILURE_MODE_THINKING_BUDGET)
        infra["thinking_budget_exhausted"] = True

    visible = _strip_thinking(final_answer_raw or "")
    if detect_repetition_loop(visible, case_id=case_id, metadata=metadata):
        if FAILURE_MODE_REPETITION_LOOP not in result.failure_modes:
            result.failure_modes.append(FAILURE_MODE_REPETITION_LOOP)
        infra["repetition_loop"] = True

    categories = [
        mode
        for mode in (
            "ollama_tool_binding_unsupported",
            "missing_expected_tool",
            "tool_required_but_not_called",
            "malformed_tool_call_syntax",
            "agent_step_limit_stub",
            FAILURE_MODE_THINKING_BUDGET,
            FAILURE_MODE_REPETITION_LOOP,
            "empty_final_answer",
        )
        if mode in result.failure_modes
    ]
    infra["categories"] = categories
    result.signals["infrastructure"] = infra


def infrastructure_counts_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Aggregate infrastructure-related failure mode counts from JSONL rows."""
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("passed"):
            continue
        for mode in row.get("failure_modes") or []:
            if mode in {
                "ollama_tool_binding_unsupported",
                "missing_expected_tool",
                "tool_required_but_not_called",
                "malformed_tool_call_syntax",
                "agent_step_limit_stub",
                FAILURE_MODE_THINKING_BUDGET,
                FAILURE_MODE_REPETITION_LOOP,
                "empty_final_answer",
            }:
                counts[mode] = counts.get(mode, 0) + 1
    return dict(sorted(counts.items()))

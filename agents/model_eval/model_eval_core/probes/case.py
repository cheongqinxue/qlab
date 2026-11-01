"""Lightweight failure-mode probes (why a case failed, not just pass/fail)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from model_eval_core.coding.exec import evaluate_coding_answer
from model_eval_core.domain.dimensions import EvalDimension, is_qualitative
from model_eval_core.probes.infrastructure import append_infrastructure_probes
from model_eval_core.text import strip_thinking

FAILURE_MODE_OLLAMA_TOOL_BINDING_UNSUPPORTED = "ollama_tool_binding_unsupported"


@dataclass
class ProbeResult:
    """Structured probe output for one eval case."""

    failure_modes: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _tool_names(messages: list[BaseMessage]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                names.append(str(call.get("name", "")))
    return names


def _final_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            if isinstance(content, str) and content.strip():
                return content
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _extract_boxed(solution: str) -> str:
    """Extract the last \\boxed{...} expression from a Hendrycks MATH solution."""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return solution.strip()
    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(solution) and depth > 0:
        if solution[pos] == "{":
            depth += 1
        elif solution[pos] == "}":
            depth -= 1
        pos += 1
    return solution[start : pos - 1].strip()


def _normalize_math(expr: str) -> str:
    """Normalize LaTeX math strings for substring matching."""
    cleaned = expr.replace("\\dfrac", "\\frac").replace("\\,", "").replace(" ", "")
    cleaned = cleaned.replace("\\$", "").replace("$", "")
    # Remove x\in prefix if present
    cleaned = re.sub(r"^x\\in", "", cleaned)
    return cleaned


def _math_matches(answer: str, target_raw: str) -> bool:
    """Return True when the normalized boxed target (or plain fraction equivalent) appears in answer."""
    boxed = _extract_boxed(target_raw)
    norm_ans = _normalize_math(answer)
    norm_box = _normalize_math(boxed)
    if norm_box and norm_box in norm_ans:
        return True
    # Check plain fraction form \frac{a}{b} -> a/b
    frac_match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", norm_box)
    if frac_match:
        plain = f"{frac_match.group(1)}/{frac_match.group(2)}"
        if plain in norm_ans:
            return True
    return False


def probe_case(
    *,
    dimension: EvalDimension,
    prompt: str,
    messages: list[BaseMessage],
    metadata: dict[str, Any],
    case_id: str = "",
) -> ProbeResult:
    """Apply dimension-aware heuristics to trace messages.

    Args:
        dimension: Eval axis for the case.
        prompt: User prompt text.
        messages: Full ReAct message list after the run.
        metadata: Curated case metadata from the benchmark row.

    Returns:
        ProbeResult with failure mode tags and diagnostic signals.
    """
    result = ProbeResult()
    answer = strip_thinking(_final_text(messages).strip())
    tools_used = _tool_names(messages)
    result.signals = {
        "tools_used": tools_used,
        "answer_chars": len(answer),
        "tool_call_count": len(tools_used),
    }

    if is_qualitative(dimension):
        result.signals["qualitative_dimension"] = True
        if metadata.get("instruction_id_list"):
            result.signals["instruction_ids"] = metadata.get("instruction_id_list")
        if not answer:
            result.failure_modes.append("empty_final_answer")
        append_infrastructure_probes(
            result,
            case_id=case_id,
            final_answer_raw=_final_text(messages),
            messages=messages,
            metadata=metadata,
        )
        result.notes = (
            "qualitative_human_review"
            if not result.failure_modes
            else "; ".join(result.failure_modes)
        )
        return result

    if not answer:
        result.failure_modes.append("empty_final_answer")

    lower = answer.lower()
    if re.search(r"\b(i think|probably|maybe|might be)\b", lower):
        result.failure_modes.append("hedged_unverified_claim")

    if dimension == EvalDimension.TOOL_USE:
        expected = metadata.get("expected_tools") or metadata.get("expected_tool")
        if expected:
            expected_list = expected if isinstance(expected, list) else [expected]
            missing = [name for name in expected_list if name not in tools_used]
            if missing:
                result.failure_modes.append("missing_expected_tool")
                result.signals["missing_tools"] = missing
        if not tools_used and re.search(r"\b(use the|call the|tool)\b", prompt.lower()):
            result.failure_modes.append("tool_required_but_not_called")
        # Detect raw XML/text tool call emission without actual tool invocation
        if not tools_used and re.search(
            r'name="(get_weather|calculator|lookup_reference)"', answer
        ):
            result.failure_modes.append("malformed_tool_call_syntax")

    if dimension == EvalDimension.MATH:
        solution = str(metadata.get("solution", ""))
        if solution and not _math_matches(answer, solution):
            result.failure_modes.append("answer_mismatch_reference")
            result.signals["expected_boxed"] = _extract_boxed(solution)

    if dimension == EvalDimension.CODING:
        entry_point = metadata.get("entry_point")
        exec_result = evaluate_coding_answer(
            prompt=prompt,
            answer=answer,
            metadata=metadata,
        )
        if exec_result is not None:
            result.signals["humaneval_exec"] = exec_result.as_dict()
            if exec_result.blocked:
                result.failure_modes.append("code_safety_blocked")
            elif not exec_result.passed:
                result.failure_modes.append("humaneval_tests_failed")
        elif entry_point and entry_point not in answer:
            result.failure_modes.append("missing_entry_point_in_answer")

    if re.search(r"need more steps to process", answer, flags=re.IGNORECASE):
        result.failure_modes.append("agent_step_limit_stub")

    if dimension == EvalDimension.REASONING:
        expected_key = str(metadata.get("answer_key", "")).strip().upper()
        if expected_key:
            match = re.search(r"final answer:\s*([A-D])\b", answer, flags=re.IGNORECASE)
            if not match or match.group(1).upper() != expected_key:
                result.failure_modes.append("answer_mismatch_reference")
                result.signals["expected_answer_key"] = expected_key

    if isinstance(messages[-1], ToolMessage) if messages else False:
        result.failure_modes.append("ended_on_tool_output")

    raw_final = _final_text(messages)
    append_infrastructure_probes(
        result,
        case_id=case_id,
        final_answer_raw=raw_final,
        messages=messages,
        metadata=metadata,
    )

    result.notes = "; ".join(result.failure_modes) if result.failure_modes else "no_probe_flags"
    return result


class ToolBindingSkippedError(Exception):
    """Internal signal when the suite skips ReAct after a prior tool-binding refusal."""


_TOOL_BINDING_MESSAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"does not support tools", re.IGNORECASE),
    re.compile(r"failed to parse grammar", re.IGNORECASE),
)


def _status_code(exc: BaseException) -> int | None:
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
    chunks: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chunks.append(str(current))
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and err.get("message"):
                chunks.append(str(err["message"]))
            elif err:
                chunks.append(str(err))
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return " ".join(chunks)


def is_ollama_tool_binding_unsupported(exc: BaseException) -> bool:
    """True when Ollama rejected tool binding for this model (HTTP 400 + known message)."""
    status = _status_code(exc)
    if status is not None and status != 400:
        return False
    text = _collect_error_text(exc)
    return any(pattern.search(text) for pattern in _TOOL_BINDING_MESSAGE_PATTERNS)


def probe_tool_binding_unsupported(
    exc: BaseException,
    *,
    skipped_without_invoke: bool = False,
) -> ProbeResult:
    """Probe output for models that cannot run Ollama tool binding."""
    signals: dict[str, Any] = {
        "tools_used": [],
        "answer_chars": 0,
        "tool_call_count": 0,
        "error_type": type(exc).__name__,
        "error_message": _collect_error_text(exc)[:500],
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

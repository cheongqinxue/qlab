"""Langfuse scores + local pass/fail for eval runs."""

from __future__ import annotations

from model_eval_core.dimensions import EvalDimension
from model_eval_core.human_review import load_human_review
from model_eval_core.probes import ProbeResult
from model_eval_core.qualitative import is_qualitative


def case_passed(
    probe: ProbeResult,
    *,
    dimension: EvalDimension | None = None,
    model_dir: str | None = None,
    case_id: str | None = None,
    max_failure_modes: int = 0,
) -> bool:
    """Return pass/fail for a case.

    Qualitative dimensions use committed human review when model_dir and case_id
    are provided; otherwise only empty answers / runtime errors fail via probes.
    """
    blocking = {
        "empty_final_answer",
        "runtime_error",
        "ollama_tool_binding_unsupported",
    }
    if any(mode in blocking for mode in probe.failure_modes):
        return False
    if dimension is not None and is_qualitative(dimension):
        if model_dir and case_id:
            review = load_human_review()
            key = (model_dir, case_id)
            if key in review:
                return review[key]
        return True
    return len(probe.failure_modes) <= max_failure_modes


def emit_langfuse_scores(
    *,
    trace_id: str | None,
    probe: ProbeResult,
    passed: bool,
    dimension: str,
    model: str,
    case_id: str,
) -> None:
    """Write eval scores to Langfuse when tracing is enabled.

    Args:
        trace_id: Langfuse trace id from the callback handler (if available).
        probe: Probe output for the case.
        passed: Overall pass boolean.
        dimension: Eval dimension string.
        model: Model name under test.
        case_id: Curated case identifier.
    """
    if not trace_id:
        return
    from agents_utils.observability.langfuse_tracing import (
        ensure_langfuse_client,
        langfuse_tracing_enabled,
    )

    if not langfuse_tracing_enabled():
        return
    ensure_langfuse_client()
    from langfuse import get_client

    client = get_client()
    client.create_score(
        trace_id=trace_id,
        name="eval_pass",
        value=1.0 if passed else 0.0,
        data_type="NUMERIC",
        comment=f"{dimension} {case_id} model={model}"[:180],
    )
    client.create_score(
        trace_id=trace_id,
        name="failure_mode_count",
        value=float(len(probe.failure_modes)),
        data_type="NUMERIC",
    )
    if probe.failure_modes:
        client.create_score(
            trace_id=trace_id,
            name="failure_modes",
            value=",".join(probe.failure_modes)[:200],
            data_type="CATEGORICAL",
        )
    infra = probe.signals.get("infrastructure") if probe.signals else None
    if isinstance(infra, dict):
        categories = infra.get("categories")
        if categories:
            client.create_score(
                trace_id=trace_id,
                name="infrastructure_categories",
                value=",".join(str(c) for c in categories)[:200],
                data_type="CATEGORICAL",
            )
        max_ct = infra.get("max_completion_tokens")
        if isinstance(max_ct, int) and max_ct > 0:
            client.create_score(
                trace_id=trace_id,
                name="max_completion_tokens",
                value=float(max_ct),
                data_type="NUMERIC",
            )

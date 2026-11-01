"""Token counts and throughput estimates from saved eval JSONL (no model re-run)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def tokens_from_messages(messages: Sequence[Mapping[str, Any]] | None) -> tuple[int, int]:
    """Sum completion and prompt tokens from serialized trace messages.

    Args:
        messages: Rows from ``responses.jsonl`` ``messages`` field.

    Returns:
        Tuple of (completion_tokens, prompt_tokens).
    """
    completion = 0
    prompt = 0
    if not messages:
        return completion, prompt
    for msg in messages:
        if msg.get("type") != "ai":
            continue
        usage = msg.get("usage")
        if not isinstance(usage, dict):
            continue
        ct = usage.get("completion_tokens")
        pt = usage.get("prompt_tokens")
        if isinstance(ct, (int, float)):
            completion += int(ct)
        if isinstance(pt, (int, float)):
            prompt += int(pt)
    return completion, prompt


def aggregate_suite_tokens(
    rows: Sequence[Mapping[str, Any]],
    *,
    wall_s: float | None,
) -> dict[str, Any]:
    """Aggregate tokens for a model folder and estimate tok/s.

    Throughput uses suite ``wall_s`` when present; otherwise sum of case ``latency_s``.

    Args:
        rows: Parsed ``responses.jsonl`` objects (may include ``messages``, ``latency_s``).
        wall_s: Suite wall clock from ``summary.json`` when available.

    Returns:
        Dict with token totals and ``completion_tok_per_s`` / ``total_tok_per_s``.
    """
    completion = 0
    prompt = 0
    latency_sum = 0.0
    cases_with_usage = 0
    for row in rows:
        comp, pr = tokens_from_messages(row.get("messages"))
        if comp or pr:
            cases_with_usage += 1
        completion += comp
        prompt += pr
        lat = row.get("latency_s")
        if isinstance(lat, (int, float)):
            latency_sum += float(lat)

    total = completion + prompt
    denom = wall_s if wall_s and wall_s > 0 else latency_sum
    completion_tps = round(completion / denom, 2) if denom > 0 and completion else None
    total_tps = round(total / denom, 2) if denom > 0 and total else None

    return {
        "completion_tokens": completion,
        "prompt_tokens": prompt,
        "total_tokens": total,
        "cases_with_usage": cases_with_usage,
        "denominator_s": round(denom, 3) if denom else None,
        "denominator_source": "wall_s" if wall_s and wall_s > 0 else "latency_sum",
        "completion_tok_per_s": completion_tps,
        "total_tok_per_s": total_tps,
    }


def aggregate_suite_tokens_from_jsonl(
    jsonl_path: Any,
    *,
    wall_s: float | None,
) -> dict[str, Any]:
    """Load JSONL from path and aggregate tokens.

    Args:
        jsonl_path: Path to ``responses.jsonl``.
        wall_s: Suite wall clock seconds.

    Returns:
        See ``aggregate_suite_tokens``.
    """
    import json
    from pathlib import Path

    path = Path(jsonl_path)
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return aggregate_suite_tokens(rows, wall_s=wall_s)

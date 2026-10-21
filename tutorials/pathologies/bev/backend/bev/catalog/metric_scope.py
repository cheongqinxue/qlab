"""Build catalog regex scope from a scoped metric family plus optional path pattern."""

from __future__ import annotations

import re
from typing import Any

_METRIC_SEGMENT_RE = re.compile(r"^[\w.-]+$")
_SCOPED_FAMILY_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def metric_family_from_key(key: str) -> str | None:
    """Return ``{section}/{metric}`` for W&B-style keys with at least two segments.

    Args:
        key: Full metric key (e.g. ``train/loss``, ``val/accuracy/layer.0``).

    Returns:
        Scoped family (e.g. ``train/loss``, ``val/accuracy``) or ``None`` when invalid.
    """
    parts = key.split("/")
    if len(parts) < 2:
        return None
    section, name = parts[0], parts[1]
    if _METRIC_SEGMENT_RE.fullmatch(section) is None:
        return None
    if _METRIC_SEGMENT_RE.fullmatch(name) is None:
        return None
    return f"{section}/{name}"


def path_suffix_from_key(key: str, metric_family: str) -> str:
    """Return the path portion after ``{section}/{metric}/`` (may be empty)."""
    prefix = f"{metric_family}/"
    if key.startswith(prefix):
        return key[len(prefix) :]
    if key == metric_family:
        return ""
    return ""


def build_scoped_pattern(metric: str, path_pattern: str | None = None) -> str:
    """Compose a full-key regex from a scoped family and optional path regex.

    Args:
        metric: Scoped family ``section/name`` (e.g. ``train/loss``).
        path_pattern: Optional regex applied to the path after ``metric/``.

    Returns:
        Anchored regex string matching catalog keys.

    Raises:
        ValueError: When ``metric`` is empty or not a valid scoped family.
    """
    stripped = metric.strip()
    if not stripped:
        raise ValueError("metric required")
    if _SCOPED_FAMILY_RE.fullmatch(stripped) is None:
        raise ValueError(f"invalid scoped metric family: {stripped}")
    base = f"^{re.escape(stripped)}"
    path = (path_pattern or "").strip()
    if not path:
        return f"{base}(?:/.*)?$"
    return f"{base}/{path}$"


def resolve_catalog_pattern(
    *,
    pattern: str | None = None,
    metric: str | None = None,
    path_pattern: str | None = None,
) -> str | None:
    """Resolve API inputs to a single catalog regex.

    Legacy ``pattern`` wins when non-empty. Otherwise ``metric`` is required.

    Returns:
        Regex string or ``None`` when nothing usable was provided.
    """
    if pattern is not None and pattern.strip():
        return pattern.strip()
    if metric is not None and metric.strip():
        return build_scoped_pattern(metric, path_pattern)
    return None


def list_metric_families(connection: Any) -> list[dict[str, Any]]:
    """List distinct scoped families with plottable key counts."""
    from bev.store.family_ops import list_metric_families as list_families_from_store

    return list_families_from_store(connection)

"""Deterministic metric key parsing for ``metrics_catalog.path_segments``."""

from __future__ import annotations

import json
import re
from typing import Any

_LAYER_RE = re.compile(r"^layer\.(\d+)$")
_LAYERS_RE = re.compile(r"^layers\.(\d+)$")
_HEAD_RE = re.compile(r"^head\.(\d+)$")


def infer_dtype(value: Any) -> str:
    """Map a parsed history value to a catalog dtype string.

    Args:
        value: JSON-decoded metric value from W&B history.

    Returns:
        One of ``float``, ``int``, ``string``, or ``null``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "int"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    return "string"


def analyze_path(key: str) -> str:
    """Build ``path_segments`` JSON for a metric key.

    Args:
        key: Full W&B metric key (e.g. ``train/grad_norm/layer.3``).

    Returns:
        JSON string with ``parts`` and ``tokens`` for UI grouping.
    """
    parts = key.split("/") if key else []
    tokens: list[dict[str, Any]] = []
    for part in parts:
        if match := _LAYER_RE.match(part):
            tokens.append(
                {
                    "kind": "layer",
                    "value": part,
                    "index": int(match.group(1)),
                }
            )
        elif match := _LAYERS_RE.match(part):
            tokens.append(
                {
                    "kind": "layer",
                    "value": part,
                    "index": int(match.group(1)),
                }
            )
        elif match := _HEAD_RE.match(part):
            tokens.append(
                {
                    "kind": "head",
                    "value": part,
                    "index": int(match.group(1)),
                }
            )
        else:
            tokens.append({"kind": "literal", "value": part})
    return json.dumps({"parts": parts, "tokens": tokens}, separators=(",", ":"))

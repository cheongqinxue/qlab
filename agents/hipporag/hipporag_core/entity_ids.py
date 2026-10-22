"""Phrase normalization and stable entity identifiers."""

from __future__ import annotations

import hashlib
import re


def normalize_phrase(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace.

    Args:
        text: Raw phrase from OpenIE or NER.

    Returns:
        Normalized phrase used as the canonical node label.
    """
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed


def entity_id(phrase: str) -> str:
    """Stable id for graph MERGE keys (upstream-style mdhash prefix).

    Args:
        phrase: Raw or normalized phrase.

    Returns:
        Id string prefixed with ``entity-``.
    """
    normalized = normalize_phrase(phrase)
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    return f"entity-{digest}"

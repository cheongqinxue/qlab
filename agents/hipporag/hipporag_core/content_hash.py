"""Content hashing for idempotent indexing."""

from __future__ import annotations

import hashlib


def passage_content_hash(text: str) -> str:
    """Hash passage body for skip-unchanged re-index.

    Args:
        text: Passage body.

    Returns:
        Hex digest of UTF-8 content.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

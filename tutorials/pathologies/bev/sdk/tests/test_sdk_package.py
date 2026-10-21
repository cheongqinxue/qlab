"""Smoke tests for the companion SDK package layout."""

from __future__ import annotations

import bev_sdk


def test_bev_sdk_importable() -> None:
    """``bev_sdk`` is importable when ``sdk/`` is on PYTHONPATH."""
    assert bev_sdk.__all__ == []

"""Synthetic W&B runs, committed data under ``fixtures/data/``, and regenerate helpers."""

from bev.fixtures.paths import (
    DATA_DIR,
    E2E_MINIMAL_SCAN_ROOT,
    LLAMA70B_UPDATE_RANK_SCAN_ROOT,
    default_playwright_scan_root,
    scan_root,
)

__all__ = [
    "DATA_DIR",
    "E2E_MINIMAL_SCAN_ROOT",
    "LLAMA70B_UPDATE_RANK_SCAN_ROOT",
    "default_playwright_scan_root",
    "scan_root",
]

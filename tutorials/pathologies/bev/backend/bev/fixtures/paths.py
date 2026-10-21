"""Canonical paths to committed offline W&B scan roots."""

from __future__ import annotations

import os
from pathlib import Path

_FIXTURES_PACKAGE = Path(__file__).resolve().parent
DATA_DIR = _FIXTURES_PACKAGE / "data"

E2E_MINIMAL_NAME = "e2e_minimal"
LLAMA70B_UPDATE_RANK_NAME = "llama70b_update_rank_scan"


def scan_root(name: str) -> Path:
    """Return ``fixtures/data/{name}`` (may or may not exist on disk)."""
    return DATA_DIR / name


def default_playwright_scan_root() -> Path:
    """Scan root for Playwright E2E (override with ``BEV_SCAN_ROOT`` or ``BEV_E2E_SCAN_ROOT``).

    Defaults to the Llama-scale ``update_rank`` fixture (same as ``run-heatmap-dev.sh``).
    Set ``BEV_E2E_SCAN_ROOT`` to ``e2e_minimal`` for fast smoke runs that need ``train/loss``.
    """
    override = os.environ.get("BEV_SCAN_ROOT") or os.environ.get("BEV_E2E_SCAN_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return scan_root(LLAMA70B_UPDATE_RANK_NAME).resolve()


E2E_MINIMAL_SCAN_ROOT = scan_root(E2E_MINIMAL_NAME)
LLAMA70B_UPDATE_RANK_SCAN_ROOT = scan_root(LLAMA70B_UPDATE_RANK_NAME)

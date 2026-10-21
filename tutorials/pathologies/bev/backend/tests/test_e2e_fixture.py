"""Tests for the offline W&B fixture used by Playwright."""

from __future__ import annotations

import re
from pathlib import Path

from bev.discovery import discover_wandb_runs
from bev.ingest.pipeline import ingest_wandb_file
from bev.store.query import list_metric_keys
from bev.fixtures.paths import E2E_MINIMAL_SCAN_ROOT
from bev.fixtures.wandb_e2e import ATTENTION_HEAD_KEYS


def test_e2e_fixture_includes_attention_head_keys(tmp_path: Path) -> None:
    scan_root = E2E_MINIMAL_SCAN_ROOT
    assert scan_root.is_dir(), f"missing committed fixture: {scan_root}"
    runs = discover_wandb_runs(scan_root)
    assert len(runs) == 1
    db = ingest_wandb_file(runs[0].wandb_file, tmp_path / "run.db")
    keys = {row["key"] for row in list_metric_keys(db.connection)}
    db.connection.close()
    for key in ATTENTION_HEAD_KEYS:
        assert key in keys
    pattern = re.compile(r"train/attention_max_weight/.+/head\.\d+$")
    assert sum(1 for key in keys if pattern.search(key)) == len(ATTENTION_HEAD_KEYS)


def test_bootstrap_e2e_regenerate(tmp_path: Path) -> None:
    """Regenerate helper still writes a valid scan root (dev workflow)."""
    from bev.fixtures.wandb_e2e import bootstrap_e2e_scan_root

    scan_root = bootstrap_e2e_scan_root(tmp_path / "scan")
    runs = discover_wandb_runs(scan_root)
    assert len(runs) == 1

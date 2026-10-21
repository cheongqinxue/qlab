"""Tests for W&B run discovery."""

from pathlib import Path

from bev.discovery import discover_wandb_runs, run_db_path


def test_run_db_path_layout(tmp_path: Path) -> None:
    store = tmp_path / "bev_store"
    assert run_db_path(store, "abc123") == store / "runs" / "abc123" / "run.db"


def test_discover_empty_dir(tmp_path: Path) -> None:
    assert discover_wandb_runs(tmp_path) == []

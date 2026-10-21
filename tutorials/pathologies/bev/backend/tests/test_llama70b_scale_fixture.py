"""Llama 70B-scale update_rank W&B fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from bev.discovery import discover_wandb_runs
from bev.ingest.pipeline import ingest_wandb_file
from bev.store.query import list_metric_keys
from bev.fixtures.wandb_llama70b_scale import (
    LLAMA70B_NUM_KV_HEADS,
    LLAMA70B_NUM_LAYERS,
    LLAMA70B_NUM_Q_HEADS,
    bootstrap_llama70b_scale_scan_root,
    expected_update_rank_key_count,
    iter_llama_update_rank_keys,
    llama70b_fixture_summary,
)


class TestLlama70bKeyCounts:
    def test_full_scale_catalog_size(self) -> None:
        expected = expected_update_rank_key_count(
            LLAMA70B_NUM_LAYERS,
            LLAMA70B_NUM_Q_HEADS,
            LLAMA70B_NUM_KV_HEADS,
        )
        assert expected == 11_762
        keys = iter_llama_update_rank_keys()
        assert len(keys) == expected

    def test_scalar_events_at_twenty_checkpoints(self) -> None:
        summary = llama70b_fixture_summary()
        assert summary["update_rank_keys"] == 11_762
        assert summary["checkpoints"] == 20
        assert summary["scalar_events"] == 235_240

    def test_mini_profile_counts(self) -> None:
        layers, q, kv = 2, 2, 2
        expected = expected_update_rank_key_count(layers, q, kv)
        assert expected == 2 + layers * ((2 * q) + (2 * kv) + 3)
        assert len(iter_llama_update_rank_keys(layers, q, kv)) == expected


class TestLlama70bFixtureIngest:
    def test_mini_offline_run_ingests(self, tmp_path: Path) -> None:
        scan_root = bootstrap_llama70b_scale_scan_root(
            tmp_path / "scan",
            num_layers=2,
            num_q_heads=2,
            num_kv_heads=2,
            steps=(0, 50),
        )
        runs = discover_wandb_runs(scan_root)
        assert len(runs) == 1
        db = ingest_wandb_file(runs[0].wandb_file, tmp_path / "run.db")
        keys = list_metric_keys(db.connection)
        db.connection.close()
        assert len(keys) == expected_update_rank_key_count(2, 2, 2)

    @pytest.mark.slow
    def test_full_catalog_ingests(self, tmp_path: Path) -> None:
        """Stress ingest: 11,762 keys × 3 checkpoint steps (trimmed for CI time)."""
        scan_root = bootstrap_llama70b_scale_scan_root(
            tmp_path / "scan",
            steps=(0, 100, 200),
        )
        runs = discover_wandb_runs(scan_root)
        db = ingest_wandb_file(runs[0].wandb_file, tmp_path / "run.db")
        keys = list_metric_keys(db.connection)
        db.connection.close()
        assert len(keys) == 11_762

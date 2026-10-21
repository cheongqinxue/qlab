"""Tests for W&B file ingest into SQLite."""

import os
import sqlite3
from pathlib import Path

import pytest

from bev.ingest.pipeline import ingest_wandb_file
from bev.store.catalog_ops import list_series


class TestBevIngest:
    def test_ingest_populates_catalog_and_series(self, minimal_wandb_run, tmp_path) -> None:
        db_path = tmp_path / "run.db"
        db = ingest_wandb_file(minimal_wandb_run, db_path)
        loss = list_series(db.connection, "train/loss")
        layer = list_series(db.connection, "train/grad_norm/layer.0")
        n_metrics = db.connection.execute("SELECT COUNT(*) FROM metrics_catalog").fetchone()[0]
        offset = db.connection.execute("SELECT byte_offset FROM ingest_state").fetchone()[0]
        store_version = db.connection.execute("SELECT store_version FROM run_meta").fetchone()[0]
        db.close()
        assert loss == [(0, 1.0), (1, 0.5)]
        assert len(layer) == 2
        assert n_metrics == 3
        assert offset > 0
        assert store_version == 1

    def test_reingest_noop_does_not_advance_store_version(self, minimal_wandb_run, tmp_path) -> None:
        db_path = tmp_path / "run.db"
        ingest_wandb_file(minimal_wandb_run, db_path).close()
        first_offset = sqlite3.connect(db_path).execute(
            "SELECT byte_offset FROM ingest_state"
        ).fetchone()[0]

        ingest_wandb_file(minimal_wandb_run, db_path).close()
        connection = sqlite3.connect(db_path)
        second_offset = connection.execute("SELECT byte_offset FROM ingest_state").fetchone()[0]
        store_version = connection.execute("SELECT store_version FROM run_meta").fetchone()[0]
        n_points = connection.execute(
            "SELECT n_points FROM metrics_catalog WHERE key = 'train/loss'"
        ).fetchone()[0]
        connection.close()
        assert second_offset == first_offset
        assert store_version == 1
        assert n_points == 2

    @pytest.mark.skipif(
        not os.environ.get("BEV_REAL_WANDB"),
        reason="Set BEV_REAL_WANDB=1 to run against pathology .data",
    )
    def test_ingest_pathology_run_fixture(self, tmp_path) -> None:
        pathology_root = Path(__file__).resolve().parents[2]
        wandb_file = (
            pathology_root
            / ".data/encoder_mlm_repeat_shift_adam_collapse/wandb/wandb/run-20261010_223356-duwjb5p5/run-duwjb5p5.wandb"
        )
        if not wandb_file.is_file():
            pytest.skip("pathology wandb fixture missing")
        db = ingest_wandb_file(wandb_file, tmp_path / "run.db")
        count = db.connection.execute("SELECT COUNT(*) FROM metrics_catalog").fetchone()[0]
        db.close()
        assert count > 10

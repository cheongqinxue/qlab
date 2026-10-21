"""Tests for run store queries."""

from bev.ingest.pipeline import ingest_wandb_file
from bev.store.query import fetch_series, list_metric_keys


class TestQuery:
    def test_catalog_without_pattern_returns_all_keys(self, minimal_wandb_run, tmp_path) -> None:
        db = ingest_wandb_file(minimal_wandb_run, tmp_path / "run.db")
        keys = list_metric_keys(db.connection)
        db.close()
        assert len(keys) > 0
        assert any(row["key"] == "train/loss" for row in keys)

    def test_catalog_empty_pattern_returns_no_keys(self, minimal_wandb_run, tmp_path) -> None:
        db = ingest_wandb_file(minimal_wandb_run, tmp_path / "run.db")
        keys = list_metric_keys(db.connection, pattern="   ")
        db.close()
        assert keys == []

    def test_catalog_pattern_filters_keys(self, minimal_wandb_run, tmp_path) -> None:
        db = ingest_wandb_file(minimal_wandb_run, tmp_path / "run.db")
        keys = list_metric_keys(db.connection, pattern=r"train/grad_norm/layer\.\d+")
        db.close()
        assert [row["key"] for row in keys] == [
            "train/grad_norm/layer.0",
            "train/grad_norm/layer.1",
        ]

    def test_fetch_series_default_step_x(self, minimal_wandb_run, tmp_path) -> None:
        db = ingest_wandb_file(minimal_wandb_run, tmp_path / "run.db")
        points = fetch_series(db.connection, "train/loss")
        db.close()
        assert points == [[0.0, 1.0], [1.0, 0.5]]

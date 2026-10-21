"""Tests for metrics_catalog population."""

import sqlite3

from bev.store.catalog_ops import ensure_metric_in_catalog, list_series, upsert_metric_value
from bev.store.schema import apply_schema


class TestMetricCatalog:
    def _connection(self, tmp_path) -> sqlite3.Connection:
        connection = sqlite3.connect(tmp_path / "run.db")
        connection.execute("PRAGMA foreign_keys = ON")
        apply_schema(connection)
        connection.execute("INSERT INTO steps (step) VALUES (1)")
        connection.execute("INSERT INTO steps (step) VALUES (2)")
        return connection

    def test_first_key_inserts_catalog_row(self, tmp_path) -> None:
        connection = self._connection(tmp_path)
        metric_id = ensure_metric_in_catalog(
            connection,
            key="train/loss",
            step=1,
            value=0.5,
        )
        row = connection.execute(
            "SELECT key, n_points, path_segments FROM metrics_catalog WHERE metric_id = ?",
            (metric_id,),
        ).fetchone()
        family = connection.execute(
            "SELECT name, plottable_count FROM metrics_family WHERE name = 'train/loss'"
        ).fetchone()
        link = connection.execute(
            "SELECT family_id FROM metrics_catalog WHERE metric_id = ?",
            (metric_id,),
        ).fetchone()
        connection.close()
        assert row[0] == "train/loss"
        assert row[1] == 0
        assert "train" in row[2]
        assert family is not None
        assert family[1] == 1
        assert link[0] is not None

    def test_n_points_increments_only_on_new_steps(self, tmp_path) -> None:
        connection = self._connection(tmp_path)
        metric_id = ensure_metric_in_catalog(
            connection,
            key="train/loss",
            step=1,
            value=1.0,
        )
        upsert_metric_value(connection, metric_id=metric_id, step=1, value=1.0)
        upsert_metric_value(connection, metric_id=metric_id, step=1, value=0.9)
        upsert_metric_value(connection, metric_id=metric_id, step=2, value=0.5)
        n_points = connection.execute(
            "SELECT n_points FROM metrics_catalog WHERE metric_id = ?",
            (metric_id,),
        ).fetchone()[0]
        series = list_series(connection, "train/loss")
        connection.close()
        assert n_points == 2
        assert series == [(1, 0.9), (2, 0.5)]

"""Migrate legacy v1 run databases to schema v2."""

from __future__ import annotations

import sqlite3

from bev.store.schema import SCHEMA_VERSION, apply_schema


def _legacy_v1_schema(connection: sqlite3.Connection) -> None:
    """Create a v1-shaped database (no ``metrics_family`` / ``family_id``)."""
    connection.executescript(
        """
        CREATE TABLE schema_migrations (
          version INTEGER PRIMARY KEY,
          applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT INTO schema_migrations (version) VALUES (1);

        CREATE TABLE metrics_catalog (
          metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
          key TEXT NOT NULL UNIQUE,
          dtype TEXT NOT NULL DEFAULT 'float',
          first_step INTEGER,
          last_step INTEGER,
          n_points INTEGER NOT NULL DEFAULT 0,
          path_segments TEXT,
          storage TEXT NOT NULL DEFAULT 'inline',
          storage_ref TEXT
        );
        CREATE INDEX idx_metrics_catalog_key ON metrics_catalog (key);

        INSERT INTO metrics_catalog (key, dtype, first_step, last_step, n_points, path_segments)
        VALUES ('train/update_rank/model.layers.0.mlp.gate_proj.weight', 'float', 0, 0, 1, '[]');
        """
    )
    connection.commit()


class TestSchemaV2Migration:
    def test_apply_schema_upgrades_legacy_db_without_error(self, tmp_path) -> None:
        db_path = tmp_path / "run.db"
        connection = sqlite3.connect(db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        _legacy_v1_schema(connection)
        connection.close()

        connection = sqlite3.connect(db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        apply_schema(connection)
        version = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        family = connection.execute(
            "SELECT name, plottable_count FROM metrics_family WHERE name = 'train/update_rank'"
        ).fetchone()
        family_id = connection.execute(
            "SELECT family_id FROM metrics_catalog LIMIT 1"
        ).fetchone()[0]
        connection.close()

        assert version == SCHEMA_VERSION == 3
        assert family is not None
        assert family[1] == 1
        assert family_id is not None

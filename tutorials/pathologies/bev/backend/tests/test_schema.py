"""Tests for BEV SQLite schema."""

import sqlite3

from bev.store.schema import SCHEMA_VERSION, apply_schema


class TestBevSchema:
    def test_apply_schema_creates_core_tables(self, tmp_path) -> None:
        db_path = tmp_path / "run.db"
        connection = sqlite3.connect(db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        apply_schema(connection)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        connection.close()
        assert "metrics_catalog" in tables
        assert "metric_values" in tables
        assert "ingest_state" in tables
        assert "metrics_family" in tables
        assert SCHEMA_VERSION == 3

    def test_schema_migration_recorded_once(self, tmp_path) -> None:
        connection = sqlite3.connect(tmp_path / "run.db")
        apply_schema(connection)
        apply_schema(connection)
        versions = connection.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        connection.close()
        assert versions[-1] == (SCHEMA_VERSION,)
        assert SCHEMA_VERSION in {row[0] for row in versions}

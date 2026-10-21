"""SQLite schema for a single W&B run store."""

from __future__ import annotations

import sqlite3

from bev.catalog.metric_scope import metric_family_from_key

SCHEMA_VERSION = 3

_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version     INTEGER PRIMARY KEY,
  applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS run_meta (
  id               INTEGER PRIMARY KEY CHECK (id = 1),
  wandb_run_id     TEXT NOT NULL UNIQUE,
  display_name     TEXT,
  project          TEXT,
  entity           TEXT,
  wandb_dir        TEXT NOT NULL,
  wandb_file       TEXT NOT NULL,
  checkpoint_dir   TEXT,
  store_version    INTEGER NOT NULL DEFAULT 0,
  created_at       TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ingest_state (
  id                 INTEGER PRIMARY KEY CHECK (id = 1),
  byte_offset        INTEGER NOT NULL DEFAULT 0,
  file_size          INTEGER NOT NULL DEFAULT 0,
  file_mtime_ns      INTEGER,
  last_wandb_step    INTEGER,
  last_ingest_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS steps (
  step          INTEGER PRIMARY KEY,
  global_step   INTEGER,
  epoch         REAL,
  logged_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_steps_global_step ON steps (global_step);

CREATE TABLE IF NOT EXISTS metrics_family (
  family_id         INTEGER PRIMARY KEY AUTOINCREMENT,
  name              TEXT NOT NULL UNIQUE,
  plottable_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metrics_catalog (
  metric_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  key             TEXT NOT NULL UNIQUE,
  dtype           TEXT NOT NULL DEFAULT 'float'
                  CHECK (dtype IN ('float', 'int', 'string', 'null')),
  first_step      INTEGER,
  last_step       INTEGER,
  n_points        INTEGER NOT NULL DEFAULT 0,
  path_segments   TEXT,
  storage         TEXT NOT NULL DEFAULT 'inline'
                  CHECK (storage IN ('inline', 'file')),
  storage_ref     TEXT,
  family_id       INTEGER REFERENCES metrics_family(family_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_catalog_key ON metrics_catalog (key);

CREATE TABLE IF NOT EXISTS metric_values (
  metric_id    INTEGER NOT NULL,
  step         INTEGER NOT NULL,
  value_real   REAL,
  value_int    INTEGER,
  value_text   TEXT,
  PRIMARY KEY (metric_id, step),
  FOREIGN KEY (metric_id) REFERENCES metrics_catalog(metric_id) ON DELETE CASCADE,
  FOREIGN KEY (step) REFERENCES steps(step) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_metric_values_step ON metric_values (step);

CREATE TABLE IF NOT EXISTS alarm_events (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  step        INTEGER NOT NULL,
  rule_id     TEXT NOT NULL,
  severity    TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
  message     TEXT NOT NULL,
  metric_key  TEXT,
  fired_at    TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (step) REFERENCES steps(step)
);
CREATE INDEX IF NOT EXISTS idx_alarm_events_step ON alarm_events (step DESC);

CREATE TABLE IF NOT EXISTS snapshot_index (
  snapshot_step     INTEGER PRIMARY KEY,
  checkpoint_label  TEXT,
  manifest_path     TEXT NOT NULL,
  indexed_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _current_schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def _ensure_catalog_family_index(connection: sqlite3.Connection) -> None:
    """Create family index when ``metrics_catalog.family_id`` exists."""
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(metrics_catalog)").fetchall()
    }
    if "family_id" in columns:
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_catalog_family_id "
            "ON metrics_catalog (family_id)"
        )


def _migrate_v2(connection: sqlite3.Connection) -> None:
    """Add ``metrics_family``, ``metrics_catalog.family_id``, and backfill from catalog."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics_family (
          family_id         INTEGER PRIMARY KEY AUTOINCREMENT,
          name              TEXT NOT NULL UNIQUE,
          plottable_count   INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(metrics_catalog)").fetchall()
    }
    if "family_id" not in columns:
        connection.execute(
            """
            ALTER TABLE metrics_catalog
            ADD COLUMN family_id INTEGER REFERENCES metrics_family(family_id) ON DELETE SET NULL
            """
        )
        _ensure_catalog_family_index(connection)

    rows = connection.execute(
        "SELECT metric_id, key, dtype, family_id FROM metrics_catalog"
    ).fetchall()
    for metric_id, key, dtype, family_id in rows:
        if family_id is not None:
            continue
        family_name = metric_family_from_key(key)
        if family_name is None:
            continue
        connection.execute(
            "INSERT OR IGNORE INTO metrics_family (name, plottable_count) VALUES (?, 0)",
            (family_name,),
        )
        family_row = connection.execute(
            "SELECT family_id FROM metrics_family WHERE name = ?",
            (family_name,),
        ).fetchone()
        if family_row is None:
            continue
        fid = int(family_row[0])
        connection.execute(
            "UPDATE metrics_catalog SET family_id = ? WHERE metric_id = ?",
            (fid, metric_id),
        )

    connection.execute("UPDATE metrics_family SET plottable_count = 0")
    plottable_rows = connection.execute(
        """
        SELECT family_id, COUNT(*) AS n
        FROM metrics_catalog
        WHERE family_id IS NOT NULL AND dtype IN ('float', 'int')
        GROUP BY family_id
        """
    ).fetchall()
    for family_id, count in plottable_rows:
        connection.execute(
            "UPDATE metrics_family SET plottable_count = ? WHERE family_id = ?",
            (int(count), int(family_id)),
        )


def _rebuild_metric_families(connection: sqlite3.Connection) -> None:
    """Recompute ``metrics_family`` and ``family_id`` from catalog keys."""
    connection.execute("UPDATE metrics_catalog SET family_id = NULL")
    connection.execute("DELETE FROM metrics_family")
    rows = connection.execute(
        "SELECT metric_id, key, dtype FROM metrics_catalog"
    ).fetchall()
    for metric_id, key, dtype in rows:
        family_name = metric_family_from_key(key)
        if family_name is None:
            continue
        connection.execute(
            "INSERT OR IGNORE INTO metrics_family (name, plottable_count) VALUES (?, 0)",
            (family_name,),
        )
        family_row = connection.execute(
            "SELECT family_id FROM metrics_family WHERE name = ?",
            (family_name,),
        ).fetchone()
        if family_row is None:
            continue
        fid = int(family_row[0])
        connection.execute(
            "UPDATE metrics_catalog SET family_id = ? WHERE metric_id = ?",
            (fid, metric_id),
        )
    connection.execute("UPDATE metrics_family SET plottable_count = 0")
    plottable_rows = connection.execute(
        """
        SELECT family_id, COUNT(*) AS n
        FROM metrics_catalog
        WHERE family_id IS NOT NULL AND dtype IN ('float', 'int')
        GROUP BY family_id
        """
    ).fetchall()
    for family_id, count in plottable_rows:
        connection.execute(
            "UPDATE metrics_family SET plottable_count = ? WHERE family_id = ?",
            (int(count), int(family_id)),
        )


def _migrate_v3(connection: sqlite3.Connection) -> None:
    """Rebuild family rows as scoped ``section/metric`` names."""
    _rebuild_metric_families(connection)


def apply_schema(connection: sqlite3.Connection) -> None:
    """Create tables and apply pending migrations.

    Args:
        connection: Open SQLite connection with foreign keys enabled.
    """
    connection.executescript(_DDL)
    version = _current_schema_version(connection)
    if version < 1:
        connection.execute(
            "INSERT INTO schema_migrations (version) VALUES (?)",
            (1,),
        )
        version = 1
    if version < 2:
        _migrate_v2(connection)
        connection.execute(
            "INSERT INTO schema_migrations (version) VALUES (?)",
            (2,),
        )
        version = 2
    if version < 3:
        _migrate_v3(connection)
        connection.execute(
            "INSERT INTO schema_migrations (version) VALUES (?)",
            (3,),
        )
    _ensure_catalog_family_index(connection)
    connection.commit()

"""Open and initialize a per-run SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from bev.store.schema import apply_schema


class RunDatabase:
    """Wrapper around ``run.db`` for ingest and queries."""

    def __init__(self, path: Path) -> None:
        """Open or create the database at ``path``."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        apply_schema(self._connection)
        self._ensure_singleton_rows()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def _ensure_singleton_rows(self) -> None:
        if self._connection.execute("SELECT COUNT(*) FROM run_meta").fetchone()[0] == 0:
            self._connection.execute(
                """
                INSERT INTO run_meta (
                  id, wandb_run_id, wandb_dir, wandb_file
                ) VALUES (1, 'pending', '', '')
                """
            )
        if self._connection.execute("SELECT COUNT(*) FROM ingest_state").fetchone()[0] == 0:
            self._connection.execute("INSERT INTO ingest_state (id) VALUES (1)")
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

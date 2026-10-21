"""Per-instance W&B scan root and SQLite store access for HTTP handlers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from bev.discovery import discover_wandb_runs, run_db_path
from bev.ingest.pipeline import ingest_wandb_file


class RunStore:
    """Resolve run databases under a central store dir and optional lazy ingest."""

    def __init__(self, scan_root: Path, store_dir: Path) -> None:
        """Bind scan and store paths for one BEV server instance.

        Args:
            scan_root: Directory tree scanned for ``run-*.wandb`` files.
            store_dir: Central SQLite store (``runs/<id>/run.db``).
        """
        self.scan_root = Path(scan_root).resolve()
        self.store_dir = Path(store_dir).resolve()
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def connection(self, run_id: str) -> sqlite3.Connection | None:
        """Open the run DB if it exists."""
        path = run_db_path(self.store_dir, run_id)
        if not path.is_file():
            return None
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        return connection

    def ensure_ingested(self, run_id: str) -> Path | None:
        """Ingest from scan root when ``run_id`` is discovered but not stored yet."""
        for run in discover_wandb_runs(self.scan_root):
            if run.wandb_run_id == run_id:
                db_path = run_db_path(self.store_dir, run_id)
                ingest_wandb_file(run.wandb_file, db_path).close()
                return db_path
        return None

    def connection_or_404(self, run_id: str) -> sqlite3.Connection:
        """Return an open connection, ingesting once if needed.

        Raises:
            LookupError: When the run is not found.
        """
        if self.connection(run_id) is None and self.ensure_ingested(run_id) is None:
            raise LookupError("run not found")
        connection = self.connection(run_id)
        if connection is None:
            raise LookupError("run not found")
        return connection

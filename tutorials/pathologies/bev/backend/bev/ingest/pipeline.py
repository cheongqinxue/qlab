"""Ingest W&B transaction logs into a :class:`~bev.store.run_db.RunDatabase`."""

from __future__ import annotations

import os
from pathlib import Path

from bev.ingest.types import HistoryEvent, RunInfo
from bev.ingest.wandb_reader import WandbRunReader
from bev.store.catalog_ops import ensure_metric_in_catalog, upsert_metric_value
from bev.store.run_db import RunDatabase


def ingest_wandb_file(
    wandb_file: Path,
    db_path: Path,
    *,
    run_info: RunInfo | None = None,
    checkpoint_dir: Path | None = None,
) -> RunDatabase:
    """Scan ``wandb_file`` from the saved byte offset and append to ``db_path``.

    Args:
        wandb_file: Path to ``run-<id>.wandb``.
        db_path: Target ``run.db`` path.
        run_info: Optional metadata; loaded from the log when omitted.
        checkpoint_dir: Optional linked checkpoint directory.

    Returns:
        Open :class:`RunDatabase` after commit.
    """
    wandb_file = Path(wandb_file)
    reader = WandbRunReader(wandb_file)
    run_info = run_info or reader.read_run_info()
    wandb_run_id = run_info.wandb_run_id if run_info else wandb_file.stem.replace("run-", "")

    db = RunDatabase(db_path)
    conn = db.connection
    stat = os.stat(wandb_file)
    offset_row = conn.execute("SELECT byte_offset FROM ingest_state WHERE id = 1").fetchone()
    start_offset = int(offset_row[0]) if offset_row else 0

    conn.execute(
        """
        UPDATE run_meta SET
          wandb_run_id = ?,
          display_name = ?,
          project = ?,
          entity = ?,
          wandb_dir = ?,
          wandb_file = ?,
          checkpoint_dir = ?,
          updated_at = datetime('now')
        WHERE id = 1
        """,
        (
            wandb_run_id,
            run_info.display_name if run_info else None,
            run_info.project if run_info else None,
            run_info.entity if run_info else None,
            str(wandb_file.parent),
            str(wandb_file),
            str(checkpoint_dir) if checkpoint_dir else None,
        ),
    )

    last_offset = start_offset
    last_step: int | None = None
    events_applied = 0
    for byte_offset, event in reader.iter_history(start_offset=start_offset):
        last_offset = byte_offset
        last_step = _apply_history_event(conn, event)
        events_applied += 1

    conn.execute(
        """
        UPDATE ingest_state SET
          byte_offset = ?,
          file_size = ?,
          file_mtime_ns = ?,
          last_wandb_step = ?,
          last_ingest_at = datetime('now')
        WHERE id = 1
        """,
        (last_offset, stat.st_size, stat.st_mtime_ns, last_step),
    )
    if events_applied > 0:
        conn.execute(
            "UPDATE run_meta SET store_version = store_version + 1, updated_at = datetime('now') WHERE id = 1"
        )
    conn.commit()
    return db


def _apply_history_event(connection, event: HistoryEvent) -> int:
    connection.execute(
        """
        INSERT INTO steps (step, global_step, epoch, logged_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(step) DO UPDATE SET
          global_step = COALESCE(excluded.global_step, steps.global_step),
          epoch = COALESCE(excluded.epoch, steps.epoch),
          logged_at = COALESCE(excluded.logged_at, steps.logged_at)
        """,
        (event.step, event.global_step, event.epoch, event.logged_at),
    )
    for key, value in event.metrics.items():
        metric_id = ensure_metric_in_catalog(
            connection,
            key=key,
            step=event.step,
            value=value,
        )
        upsert_metric_value(connection, metric_id=metric_id, step=event.step, value=value)
    return event.step

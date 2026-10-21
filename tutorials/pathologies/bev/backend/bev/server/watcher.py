"""Background tail of growing ``.wandb`` files."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from bev.discovery import discover_wandb_runs, run_db_path
from bev.ingest.pipeline import ingest_wandb_file

logger = logging.getLogger(__name__)


class _WandbTailHandler(FileSystemEventHandler):
    def __init__(self, store_dir: Path) -> None:
        self._store_dir = store_dir
        self._lock = threading.Lock()

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix != ".wandb" or not path.name.startswith("run-"):
            return
        run_id = path.stem.replace("run-", "", 1)
        db_path = run_db_path(self._store_dir, run_id)
        with self._lock:
            try:
                ingest_wandb_file(path, db_path).close()
            except Exception:
                logger.exception("Failed to ingest %s", path)


def start_wandb_watcher(scan_root: Path, store_dir: Path) -> Observer:
    """Watch ``scan_root`` for ``.wandb`` changes and re-ingest into ``store_dir``.

    Args:
        scan_root: Discovery root passed to :func:`~bev.discovery.discover_wandb_runs`.
        store_dir: Central BEV SQLite store for this server instance.

    Returns:
        Started :class:`watchdog.observers.Observer` (call ``stop()`` on shutdown).
    """
    handler = _WandbTailHandler(store_dir)
    observer = Observer()
    runs = discover_wandb_runs(scan_root)
    watched = {run.run_dir for run in runs}
    if not watched:
        watched.add(Path(scan_root).resolve())
    for directory in watched:
        observer.schedule(handler, str(directory), recursive=False)
    observer.start()
    return observer


def ingest_all(scan_root: Path, store_dir: Path) -> None:
    """Ingest every discovered run under ``scan_root`` into ``store_dir``."""
    for run in discover_wandb_runs(scan_root):
        db_path = run_db_path(store_dir, run.wandb_run_id)
        ingest_wandb_file(run.wandb_file, db_path).close()

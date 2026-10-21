"""Process logging to stdout and an optional on-disk directory."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_LOG_FILE_NAME = "agent.log"
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024


class _SingleFileSizeCapHandler(RotatingFileHandler):
    """``RotatingFileHandler`` that truncates the file when ``backupCount`` is 0."""

    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.backupCount > 0:
            super().doRollover()
            return
        with open(self.baseFilename, "w", encoding=self.encoding):
            pass
        if not self.delay:
            self.stream = self._open()


def configure_agent_logging() -> Path | None:
    """Configure the root logger with stdout and optional file output.

    When ``AGENT_LOG_DIR`` is set, writes ``agent.log`` under that directory.
    Default retention is one file capped at 5MB (``AGENT_LOG_MAX_BYTES``).

    Args:
        None.

    Returns:
        Path to the active log file, or None when file logging is disabled.
    """
    level_name = os.environ.get("AGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = os.environ.get("AGENT_LOG_DIR", "").strip()

    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT)

    if not _has_stream_handler(root):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    log_file: Path | None = None
    if log_dir:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        log_file = directory / _LOG_FILE_NAME
        if not _has_file_handler(root, log_file):
            max_bytes = int(
                os.environ.get("AGENT_LOG_MAX_BYTES", str(_DEFAULT_MAX_BYTES)),
            )
            backup_count = int(os.environ.get("AGENT_LOG_BACKUP_COUNT", "0"))
            file_handler = _SingleFileSizeCapHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(logger_name).propagate = True

    return log_file


def _has_stream_handler(root: logging.Logger) -> bool:
    return any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in root.handlers
    )


def _has_file_handler(root: logging.Logger, log_file: Path) -> bool:
    target = str(log_file.resolve())
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            base = getattr(handler, "baseFilename", None)
            if base and str(Path(base).resolve()) == target:
                return True
    return False

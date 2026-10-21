"""BEV instance paths (central store, separate from W&B log layout on disk)."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_STORE_DIR = Path.home() / ".bev" / "store"
DEFAULT_ACCOUNTS_DB = Path.home() / ".bev" / "accounts.db"


def product_root() -> Path:
    """Return the BEV product root (``backend/``, ``webui/``, ``static/``, ``e2e/`` siblings)."""
    return Path(__file__).resolve().parent.parent.parent


def static_dir() -> Path:
    """Directory for the production Vite build (``webui`` → ``../static``)."""
    return product_root() / "static"


def resolve_store_dir(explicit: Path | str | None = None) -> Path:
    """Return the SQLite store root for this BEV server instance.

    Precedence: CLI ``--store-dir`` > ``BEV_STORE_DIR`` env > default ``~/.bev/store``.

    Args:
        explicit: Value from ``bev serve --store-dir``, if any.

    Returns:
        Absolute path; directory is not created by this helper.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if env := os.environ.get("BEV_STORE_DIR"):
        return Path(env).expanduser().resolve()
    return DEFAULT_STORE_DIR.expanduser().resolve()


def resolve_accounts_db_path(explicit: Path | str | None = None) -> Path:
    """Return the SQLite path for users and workspace preferences.

    Precedence: explicit arg > ``BEV_ACCOUNTS_DB`` env > ``~/.bev/accounts.db``.

    Args:
        explicit: Override for tests or embedded deployments.

    Returns:
        Absolute path; parent directory is not created by this helper.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if env := os.environ.get("BEV_ACCOUNTS_DB"):
        return Path(env).expanduser().resolve()
    return DEFAULT_ACCOUNTS_DB.expanduser().resolve()

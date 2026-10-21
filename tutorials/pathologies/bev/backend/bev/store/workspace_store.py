"""Workspace layout for the active user and scan root."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from bev.store.account_store import AccountStore, DEFAULT_ADMIN_USERNAME, UserRecord

WORKSPACE_DB_NAME = "workspace.db"


def workspace_db_path(store_dir: Path) -> Path:
    """Return the legacy workspace SQLite path under ``store_dir``."""
    return Path(store_dir).resolve() / WORKSPACE_DB_NAME


def _load_legacy_sections(store_dir: Path) -> list[dict[str, Any]]:
    """Read pre-accounts ``workspace_document`` if present."""
    path = workspace_db_path(store_dir)
    if not path.is_file():
        return []
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT sections_json FROM workspace_document WHERE id = 1"
        ).fetchone()
        if row is None:
            return []
        payload = json.loads(row[0])
        return payload if isinstance(payload, list) else []
    except sqlite3.Error:
        return []
    finally:
        connection.close()


class WorkspaceStore:
    """Read/write workspace JSON for the implicit session user and scan root."""

    def __init__(
        self,
        account_store: AccountStore,
        scan_root: Path,
        store_dir: Path,
        user: UserRecord,
    ) -> None:
        """Bind workspace persistence to one server instance.

        Args:
            account_store: Shared accounts database.
            scan_root: ``bev serve --root`` path (workspace identity).
            store_dir: Ingest store (legacy ``workspace.db`` migration only).
            user: Active user (bootstrap admin until login exists).
        """
        self._accounts = account_store
        self._scan_root = Path(scan_root).resolve()
        self._store_dir = Path(store_dir).resolve()
        self._user = user

    @property
    def scan_root(self) -> Path:
        """W&B directory this workspace is tied to."""
        return self._scan_root

    @property
    def username(self) -> str:
        """Active username (no auth UI yet)."""
        return self._user.username

    @property
    def user_role(self) -> str:
        """Active user role."""
        return self._user.role

    def load_sections(self) -> list[dict[str, Any]]:
        """Return stored workspace sections, migrating legacy data once if needed."""
        sections = self._accounts.load_workspace_sections(self._user.id, self._scan_root)
        if sections:
            return sections
        legacy = _load_legacy_sections(self._store_dir)
        if legacy:
            self._accounts.save_workspace_sections(self._user.id, self._scan_root, legacy)
            return legacy
        return []

    def save_sections(self, sections: list[dict[str, Any]]) -> None:
        """Replace workspace sections for the active user and scan root."""
        self._accounts.save_workspace_sections(self._user.id, self._scan_root, sections)


def bootstrap_workspace_store(
    account_store: AccountStore,
    scan_root: Path,
    store_dir: Path,
    *,
    admin_username: str = DEFAULT_ADMIN_USERNAME,
) -> WorkspaceStore:
    """Create a workspace store for the default admin session.

    Args:
        account_store: Accounts database.
        scan_root: W&B scan root.
        store_dir: Ingest store directory.
        admin_username: User loaded without login (default ``admin``).

    Returns:
        Configured :class:`WorkspaceStore`.
    """
    user = account_store.ensure_default_admin(username=admin_username)
    account_store.touch_scan_root(scan_root)
    return WorkspaceStore(account_store, scan_root, store_dir, user)

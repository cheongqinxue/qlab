"""User accounts, scan roots, and per-user workspace configuration."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bev.config import resolve_accounts_db_path

ACCOUNTS_SCHEMA_VERSION = 1
DEFAULT_ADMIN_USERNAME = "admin"
PBKDF2_ITERATIONS = 600_000


@dataclass(frozen=True)
class UserRecord:
    """Row from ``users``."""

    id: int
    username: str
    role: str


def hash_password(password: str) -> str:
    """Return a stored credential string for ``password`` (PBKDF2-SHA256).

    Args:
        password: Plain text (never log or persist outside this format).

    Returns:
        Encoded hash ``pbkdf2_sha256$iterations$salt_hex$digest_hex``.
    """
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check ``password`` against a value from :func:`hash_password`.

    Args:
        password: Candidate plain text.
        stored: Stored credential string.

    Returns:
        True when the password matches.
    """
    try:
        scheme, iter_str, salt, digest_hex = stored.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iter_str)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        )
        return secrets.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _apply_accounts_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version     INTEGER PRIMARY KEY,
          applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS users (
          user_id         INTEGER PRIMARY KEY,
          username        TEXT NOT NULL UNIQUE,
          password_hash   TEXT NOT NULL,
          role            TEXT NOT NULL DEFAULT 'user',
          created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS scan_roots (
          scan_root_id    INTEGER PRIMARY KEY,
          root_path       TEXT NOT NULL UNIQUE,
          first_opened_at TEXT NOT NULL DEFAULT (datetime('now')),
          last_opened_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS user_workspace_configs (
          user_id         INTEGER NOT NULL REFERENCES users(user_id),
          scan_root_id    INTEGER NOT NULL REFERENCES scan_roots(scan_root_id),
          sections_json   TEXT NOT NULL DEFAULT '[]',
          updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
          PRIMARY KEY (user_id, scan_root_id)
        );

        CREATE INDEX IF NOT EXISTS idx_scan_roots_last_opened
          ON scan_roots(last_opened_at DESC);
        """
    )
    row = connection.execute(
        "SELECT version FROM schema_migrations WHERE version = ?",
        (ACCOUNTS_SCHEMA_VERSION,),
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO schema_migrations (version) VALUES (?)",
            (ACCOUNTS_SCHEMA_VERSION,),
        )
    connection.commit()


class AccountStore:
    """SQLite store for users and workspace layout keyed by scan root."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Open or create the accounts database.

        Args:
            db_path: Override path (default :func:`~bev.config.resolve_accounts_db_path`).
        """
        self._path = Path(db_path or resolve_accounts_db_path()).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path)
        try:
            _apply_accounts_schema(connection)
        finally:
            connection.close()

    def ensure_default_admin(
        self,
        *,
        username: str = DEFAULT_ADMIN_USERNAME,
        password: str | None = None,
        role: str = "admin",
    ) -> UserRecord:
        """Ensure the bootstrap admin user exists (no login UI yet).

        Args:
            username: Default ``admin``.
            password: Plain text; defaults to ``BEV_ADMIN_PASSWORD`` env or ``password``.
            role: Stored role (``admin`` for the bootstrap account).

        Returns:
            Existing or newly created user row.
        """
        plain = password if password is not None else os.environ.get("BEV_ADMIN_PASSWORD", "password")
        connection = sqlite3.connect(self._path)
        try:
            row = connection.execute(
                "SELECT user_id, username, role FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if row is not None:
                return UserRecord(id=row[0], username=row[1], role=row[2])
            connection.execute(
                """
                INSERT INTO users (username, password_hash, role)
                VALUES (?, ?, ?)
                """,
                (username, hash_password(plain), role),
            )
            connection.commit()
            user_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
            return UserRecord(id=user_id, username=username, role=role)
        finally:
            connection.close()

    def get_user_by_username(self, username: str) -> UserRecord | None:
        """Return a user by username or ``None``."""
        connection = sqlite3.connect(self._path)
        try:
            row = connection.execute(
                "SELECT user_id, username, role FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if row is None:
                return None
            return UserRecord(id=row[0], username=row[1], role=row[2])
        finally:
            connection.close()

    def touch_scan_root(self, scan_root: Path | str) -> int:
        """Register or refresh a W&B scan root path.

        Args:
            scan_root: Absolute path passed to ``bev serve --root``.

        Returns:
            ``scan_root_id`` primary key.
        """
        root_path = str(Path(scan_root).expanduser().resolve())
        connection = sqlite3.connect(self._path)
        try:
            connection.execute(
                """
                INSERT INTO scan_roots (root_path)
                VALUES (?)
                ON CONFLICT(root_path) DO UPDATE SET last_opened_at = datetime('now')
                """,
                (root_path,),
            )
            connection.commit()
            row = connection.execute(
                "SELECT scan_root_id FROM scan_roots WHERE root_path = ?",
                (root_path,),
            ).fetchone()
            assert row is not None
            return int(row[0])
        finally:
            connection.close()

    def load_workspace_sections(self, user_id: int, scan_root: Path | str) -> list[dict[str, Any]]:
        """Load panel layout for one user and scan root.

        Args:
            user_id: Owner of the configuration.
            scan_root: W&B scan root path.

        Returns:
            Section list (empty when unset).
        """
        root_path = str(Path(scan_root).expanduser().resolve())
        connection = sqlite3.connect(self._path)
        try:
            row = connection.execute(
                """
                SELECT uwc.sections_json
                FROM user_workspace_configs uwc
                JOIN scan_roots sr ON sr.scan_root_id = uwc.scan_root_id
                WHERE uwc.user_id = ? AND sr.root_path = ?
                """,
                (user_id, root_path),
            ).fetchone()
            if row is None:
                return []
            payload = json.loads(row[0])
            return payload if isinstance(payload, list) else []
        finally:
            connection.close()

    def save_workspace_sections(
        self,
        user_id: int,
        scan_root: Path | str,
        sections: list[dict[str, Any]],
    ) -> None:
        """Replace workspace sections for one user and scan root."""
        scan_root_id = self.touch_scan_root(scan_root)
        connection = sqlite3.connect(self._path)
        try:
            connection.execute(
                """
                INSERT INTO user_workspace_configs (user_id, scan_root_id, sections_json, updated_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(user_id, scan_root_id) DO UPDATE SET
                  sections_json = excluded.sections_json,
                  updated_at = datetime('now')
                """,
                (user_id, scan_root_id, json.dumps(sections)),
            )
            connection.commit()
        finally:
            connection.close()

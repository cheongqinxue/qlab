"""Accounts database and per-root workspace isolation."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from bev.server.app import create_app
from bev.store.account_store import AccountStore, verify_password


@pytest.fixture
def accounts_db(tmp_path: Path) -> Path:
    return tmp_path / "accounts.db"


def test_default_admin_password_hash(accounts_db: Path) -> None:
    store = AccountStore(accounts_db)
    user = store.ensure_default_admin(password="password")
    assert user.username == "admin"
    assert user.role == "admin"
    connection_row = _password_hash(accounts_db, "admin")
    assert verify_password("password", connection_row)
    assert not verify_password("wrong", connection_row)


def test_workspace_isolated_by_scan_root(accounts_db: Path) -> None:
    store = AccountStore(accounts_db)
    admin = store.ensure_default_admin(password="secret")
    root_a = Path("/tmp/bev-root-a").resolve()
    root_b = Path("/tmp/bev-root-b").resolve()
    store.save_workspace_sections(admin.id, root_a, [{"id": "a", "panels": []}])
    store.save_workspace_sections(admin.id, root_b, [{"id": "b", "panels": []}])
    assert store.load_workspace_sections(admin.id, root_a)[0]["id"] == "a"
    assert store.load_workspace_sections(admin.id, root_b)[0]["id"] == "b"


def test_workspace_api_includes_scan_root(minimal_wandb_run: Path, tmp_path: Path) -> None:
    root = minimal_wandb_run.parent.parent
    store_dir = tmp_path / "bev-store"
    accounts_db = tmp_path / "accounts.db"
    client = TestClient(create_app(root, store_dir, accounts_db=accounts_db))
    response = client.get("/api/workspace")
    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "admin"
    assert body["scan_root"] == str(root.resolve())


def _password_hash(db_path: Path, username: str) -> str:
    import sqlite3

    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        assert row is not None
        return str(row[0])
    finally:
        connection.close()

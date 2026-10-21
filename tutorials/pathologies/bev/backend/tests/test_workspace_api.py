"""Workspace document API tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from bev.server.app import create_app


@pytest.fixture
def api_client(minimal_wandb_run: Path, tmp_path: Path) -> TestClient:
    root = minimal_wandb_run.parent.parent
    store_dir = tmp_path / "bev-store"
    accounts_db = tmp_path / "accounts.db"
    return TestClient(create_app(root, store_dir, accounts_db=accounts_db))


def test_workspace_put_and_get(api_client: TestClient) -> None:
    sections = [
        {
            "id": "section-1",
            "name": "Metrics",
            "isOpen": True,
            "columns": 2,
            "panels": [],
        }
    ]
    put = api_client.put("/api/workspace", json={"sections": sections})
    assert put.status_code == 200
    assert put.json()["sections"] == sections
    get = api_client.get("/api/workspace")
    assert get.status_code == 200
    body = get.json()
    assert body["sections"] == sections
    assert body["username"] == "admin"
    assert body["scan_root"]

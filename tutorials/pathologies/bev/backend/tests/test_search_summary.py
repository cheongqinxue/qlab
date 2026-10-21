"""Collapsed search summary trees."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from bev.catalog.search_summary import build_search_summary_tree
from bev.server.app import create_app


@pytest.fixture
def api_client(minimal_wandb_run: Path, tmp_path: Path) -> TestClient:
    root = minimal_wandb_run.parent.parent
    store_dir = tmp_path / "bev-store"
    accounts_db = tmp_path / "accounts.db"
    return TestClient(create_app(root, store_dir, accounts_db=accounts_db))


def test_summary_collapses_layer_indices() -> None:
    keys = [f"train/update_rank/model.layers.{layer}/head.{head}" for layer in range(4) for head in range(2)]
    payload = build_search_summary_tree(keys, max_nodes=200)
    assert payload["key_count"] == len(keys)
    serialized = str(payload)
    assert "0..3" in serialized or "layers" in serialized


def test_metrics_search_summary_query(api_client) -> None:
    client = api_client
    runs = client.get("/api/runs").json()
    run_id = runs[0]["wandb_run_id"]
    plain = client.get(
        f"/api/runs/{run_id}/metrics/search",
        params={"pattern": r"train/", "limit": 2},
    )
    assert plain.status_code == 200
    assert plain.json().get("summary") is None

    rich = client.get(
        f"/api/runs/{run_id}/metrics/search",
        params={"pattern": r"train/", "summary": 1},
    )
    assert rich.status_code == 200
    body = rich.json()
    assert body["match_count"] >= 1
    assert body["summary"] is not None
    assert body["summary"]["key_count"] == body["match_count"]

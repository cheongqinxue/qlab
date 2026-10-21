"""SQL pooling for batch series reads."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from bev.fixtures.paths import E2E_MINIMAL_SCAN_ROOT
from bev.ingest.pipeline import ingest_wandb_file
from bev.server.app import create_app
from bev.store.catalog_query import fetch_series_batch_for_keys
from bev.store.query import fetch_pooled_series_for_keys


@pytest.fixture
def e2e_api_client(tmp_path: Path) -> TestClient:
    assert E2E_MINIMAL_SCAN_ROOT.is_dir()
    return TestClient(create_app(E2E_MINIMAL_SCAN_ROOT, tmp_path / "store"))


def test_fetch_pooled_mean_across_layers(minimal_wandb_run, tmp_path) -> None:
    db = ingest_wandb_file(minimal_wandb_run, tmp_path / "run.db")
    keys = ["train/grad_norm/layer.0", "train/grad_norm/layer.1"]
    pooled = fetch_pooled_series_for_keys(db.connection, keys, "mean")
    db.close()
    assert pooled == [[0.0, 0.5], [1.0, 0.175]]


def test_batch_with_pool_op_returns_pooled_only(minimal_wandb_run, tmp_path) -> None:
    db = ingest_wandb_file(minimal_wandb_run, tmp_path / "run.db")
    payload = fetch_series_batch_for_keys(
        db.connection,
        ["train/grad_norm/layer.0", "train/grad_norm/layer.1"],
        pool_op="max",
        max_points=2000,
    )
    db.close()
    assert payload["pool_op"] == "max"
    assert payload["series"] == {}
    assert payload["pooled_series"] == [[0.0, 0.5], [1.0, 0.25]]


def test_batch_invalid_pool_op_http(e2e_api_client: TestClient) -> None:
    runs = e2e_api_client.get("/api/runs").json()
    run_id = runs[0]["wandb_run_id"]
    response = e2e_api_client.post(
        f"/api/runs/{run_id}/series/batch",
        json={"keys": ["train/loss"], "pool_op": "p95"},
    )
    assert response.status_code == 400

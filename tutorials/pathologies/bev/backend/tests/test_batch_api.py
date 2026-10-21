"""Tests for catalog preview and series batch APIs."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from bev.fixtures.paths import E2E_MINIMAL_SCAN_ROOT
from bev.server.app import create_app


@pytest.fixture
def api_client(minimal_wandb_run: Path, tmp_path: Path) -> TestClient:
    root = minimal_wandb_run.parent.parent
    store_dir = tmp_path / "bev-store"
    return TestClient(create_app(root, store_dir))


@pytest.fixture
def e2e_api_client(tmp_path: Path) -> TestClient:
    assert E2E_MINIMAL_SCAN_ROOT.is_dir()
    store_dir = tmp_path / "bev-store"
    return TestClient(create_app(E2E_MINIMAL_SCAN_ROOT, store_dir))


def test_catalog_rejects_filters(api_client: TestClient) -> None:
    runs = api_client.get("/api/runs").json()
    run_id = runs[0]["wandb_run_id"]
    response = api_client.get(
        f"/api/runs/{run_id}/catalog",
        params={"pattern": r"train/", "limit": 2},
    )
    assert response.status_code == 400


def test_metrics_search_preview_limit(api_client: TestClient) -> None:
    runs = api_client.get("/api/runs").json()
    run_id = runs[0]["wandb_run_id"]
    response = api_client.get(
        f"/api/runs/{run_id}/metrics/search",
        params={"pattern": r"train/", "limit": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["match_count"] >= 2
    assert len(body["metrics"]) == 2
    assert body["truncated"] is True


def test_series_batch(api_client: TestClient) -> None:
    runs = api_client.get("/api/runs").json()
    run_id = runs[0]["wandb_run_id"]
    response = api_client.post(
        f"/api/runs/{run_id}/series/batch",
        json={"keys": ["train/loss"], "max_keys": 10},
    )
    assert response.status_code == 200
    body = response.json()
    assert "train/loss" in body["series"]
    assert len(body["series"]["train/loss"]) >= 1


def test_series_batch_requires_keys(api_client: TestClient) -> None:
    runs = api_client.get("/api/runs").json()
    run_id = runs[0]["wandb_run_id"]
    response = api_client.post(
        f"/api/runs/{run_id}/series/batch",
        json={"pattern": r"train/loss", "max_keys": 10},
    )
    assert response.status_code in (400, 422)


def test_series_batch_explicit_keys(api_client: TestClient) -> None:
    runs = api_client.get("/api/runs").json()
    run_id = runs[0]["wandb_run_id"]
    response = api_client.post(
        f"/api/runs/{run_id}/series/batch",
        json={"keys": ["train/loss"], "max_keys": 10},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["keys"] == ["train/loss"]
    assert "train/loss" in body["series"]


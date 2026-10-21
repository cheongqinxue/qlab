"""Metric family + path pattern scoping."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from bev.catalog.metric_scope import build_scoped_pattern, metric_family_from_key
from bev.fixtures.paths import E2E_MINIMAL_SCAN_ROOT
from bev.server.app import create_app


def test_metric_family_from_key() -> None:
    assert metric_family_from_key("train/loss") == "train/loss"
    assert metric_family_from_key("train/grad_norm/layer.0") == "train/grad_norm"
    assert metric_family_from_key("train/attention_max_weight/bert/layer.0/head.0") == (
        "train/attention_max_weight"
    )
    assert metric_family_from_key("val/loss") == "val/loss"
    assert metric_family_from_key("train") is None


def test_build_scoped_pattern() -> None:
    assert build_scoped_pattern("train/loss") == r"^train/loss(?:/.*)?$"
    assert build_scoped_pattern("train/grad_norm", r"layer\.\d+") == (
        r"^train/grad_norm/layer\.\d+$"
    )
    assert build_scoped_pattern("val/loss") == r"^val/loss(?:/.*)?$"


@pytest.fixture
def e2e_api_client(tmp_path: Path) -> TestClient:
    assert E2E_MINIMAL_SCAN_ROOT.is_dir()
    return TestClient(create_app(E2E_MINIMAL_SCAN_ROOT, tmp_path / "store"))


def test_series_batch_explicit_keys_scope(e2e_api_client: TestClient) -> None:
    runs = e2e_api_client.get("/api/runs").json()
    run_id = runs[0]["wandb_run_id"]
    search = e2e_api_client.get(
        f"/api/runs/{run_id}/metrics/search",
        params={
            "metric": "train/attention_max_weight",
            "path_pattern": r".+/head\.\d+$",
        },
    )
    assert search.status_code == 200
    keys = [
        row["key"]
        for row in search.json()["metrics"]
        if row.get("dtype") in ("float", "int")
    ][:32]
    assert keys
    response = e2e_api_client.post(
        f"/api/runs/{run_id}/series/batch",
        json={"keys": keys, "max_keys": 32, "scope": "layer.0"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "layer.0"
    assert body["series"]
    assert all("layer.0" in key for key in body["series"])


def test_metrics_search_metric_scope(e2e_api_client: TestClient) -> None:
    runs = e2e_api_client.get("/api/runs").json()
    run_id = runs[0]["wandb_run_id"]
    response = e2e_api_client.get(
        f"/api/runs/{run_id}/metrics/search",
        params={
            "metric": "train/attention_max_weight",
            "path_pattern": r".+/head\.\d+$",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["match_count"] >= 1
    assert all("attention_max_weight" in row["key"] for row in body["metrics"])


def test_metric_families_http(e2e_api_client: TestClient) -> None:
    runs = e2e_api_client.get("/api/runs").json()
    run_id = runs[0]["wandb_run_id"]
    response = e2e_api_client.get(f"/api/runs/{run_id}/metric-families")
    assert response.status_code == 200
    families = response.json()["families"]
    ids = {row["id"] for row in families}
    assert "train/attention_max_weight" in ids

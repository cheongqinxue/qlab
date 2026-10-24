"""HTTP API smoke tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _load_cove_app():
    """Import CoVe ``main:app`` without colliding with other experiments."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("cove_main", root / "main.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.app


@pytest.fixture
def api_client() -> TestClient:
    """CoVe app with lifespan (overrides conftest client for explicit main load)."""
    with TestClient(_load_cove_app()) as client:
        yield client


class TestCoVeApi:
    """Health and verify routes."""

    def test_health(self, api_client: TestClient) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["experiment"] == "cove"
        assert body["agents_mode"] == "fixture"

    def test_bootstrap_toy(self, api_client: TestClient) -> None:
        response = api_client.post("/eval/bootstrap-toy")
        assert response.status_code == 200
        body = response.json()
        assert body["query_id"] == "mexican_american_war"
        assert "1848" in body["final_response"]

    def test_verify_with_query_id(self, api_client: TestClient) -> None:
        response = api_client.post(
            "/verify",
            json={"query_id": "mexican_american_war"},
        )
        assert response.status_code == 200
        assert response.json()["execution_mode"] == "factored"

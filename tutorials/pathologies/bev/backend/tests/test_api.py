"""HTTP API smoke tests."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from bev.server.app import create_app


@pytest.fixture
def api_client(minimal_wandb_run: Path, tmp_path: Path) -> TestClient:
    """Serve API rooted at the parent of the generated offline run."""
    root = minimal_wandb_run.parent.parent
    store_dir = tmp_path / "bev-store"
    return TestClient(create_app(root, store_dir))


def test_list_runs_and_series(api_client: TestClient) -> None:
    runs = api_client.get("/api/runs")
    assert runs.status_code == 200
    body = runs.json()
    assert len(body) >= 1
    run_id = body[0]["wandb_run_id"]

    manifest = api_client.get(f"/api/runs/{run_id}/manifest")
    assert manifest.status_code == 200
    assert manifest.json()["store_version"] >= 1

    series = api_client.get(f"/api/runs/{run_id}/series", params={"key": "train/loss"})
    assert series.status_code == 200
    assert len(series.json()["points"]) >= 1

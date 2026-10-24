"""FastAPI route tests."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _load_path_setup():
    """Load ``tests/_path_setup.py`` without importing as ``tests._path_setup``."""
    path = Path(__file__).resolve().parent / "_path_setup.py"
    spec = importlib.util.spec_from_file_location("hmem_path_setup", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_hmem_app():
    """Import H-MEM ``main:app`` without colliding with other experiments."""
    path_setup = _load_path_setup()
    path_setup.ensure_hmem_paths()
    path_setup.purge_shared_agent_modules()
    root = Path(__file__).resolve().parents[1]
    os.environ["HMEM_CONFIG"] = str(root / "config" / "default.json")
    from server.settings import get_settings

    get_settings.cache_clear()
    spec = importlib.util.spec_from_file_location("hmem_main", root / "main.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.app


@pytest.fixture
def api_client() -> TestClient:
    """H-MEM app with in-memory hybrid store."""
    with TestClient(_load_hmem_app()) as client:
        yield client


class TestHealth:
    """Health endpoint."""

    def test_health_ok(self, api_client: TestClient) -> None:
        """Health returns experiment id and hybrid counts."""
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["experiment"] == "hmem"
        assert body["agents_mode"] == "fixture"


class TestToyFlow:
    """Bootstrap and ask flow."""

    def test_bootstrap_and_ask(self, api_client: TestClient) -> None:
        """Toy corpus indexes and fixture ask returns Mochi."""
        boot = api_client.post("/corpus/bootstrap-toy")
        assert boot.status_code == 200
        assert boot.json()["fragments"] == 3

        ask = api_client.post("/ask", json={"query": "What is the cat's name?"})
        assert ask.status_code == 200
        assert ask.json()["answer"] == "Mochi"

"""FastAPI route tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _load_path_setup():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "_path_setup.py"
    spec = importlib.util.spec_from_file_location("mgr_path_setup", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def api_client() -> TestClient:
    """MemGraphRAG app with in-process memory."""
    path_setup = _load_path_setup()
    path_setup.ensure_memgraphrag_paths()
    path_setup.purge_shared_agent_modules()
    from main import app

    with TestClient(app) as client:
        yield client


class TestHealth:
    """Health endpoint."""

    def test_health_ok(self, api_client: TestClient) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["experiment"] == "memgraphrag"
        assert "ontology" in body["memory"]


class TestReset:
    """Reset endpoint."""

    def test_reset_clears_memory(self, api_client: TestClient) -> None:
        from memgraphrag_core.models import PassageRecord

        mem = api_client.app.state.runtime.memory  # type: ignore[attr-defined]
        mem.layers.add_passage(PassageRecord(id="p1", text="seed"))
        assert mem.passage_count == 1
        response = api_client.post("/reset")
        assert response.status_code == 200
        assert response.json()["status"] == "cleared"
        assert mem.passage_count == 0


class TestCorpusIndexRetrieve:
    """Toy bootstrap and retrieval flow."""

    def test_bootstrap_and_retrieve(self, api_client: TestClient) -> None:
        boot = api_client.post("/corpus/bootstrap-toy")
        assert boot.status_code == 200
        body = boot.json()
        assert body["facts"] == 1
        assert body["conflicts_resolved"] >= 1

        response = api_client.post(
            "/retrieve",
            json={"query": "When was Isaac Newton born?", "top_k": 2},
        )
        assert response.status_code == 200
        hits = response.json()["hits"]
        assert len(hits) >= 1
        assert "1642" in hits[0]["content"] or "Newton" in hits[0]["content"]

    def test_passage_and_index(self, api_client: TestClient) -> None:
        api_client.post("/reset")
        api_client.post(
            "/corpus/passages",
            json={"id": "p1", "text": "Paris is the capital of France."},
        )
        index_resp = api_client.post("/index")
        assert index_resp.status_code == 200
        assert index_resp.json()["passages"] == 1

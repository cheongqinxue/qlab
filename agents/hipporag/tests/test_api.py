"""FastAPI routes (in-memory backend)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _load_hipporag_app():
    """Import HippoRAG ``main:app`` without colliding with other experiments."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("hipporag_main", root / "main.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.app


@pytest.fixture
def api_client() -> TestClient:
    """App with memory graph store (no Neo4j)."""
    with TestClient(_load_hipporag_app()) as client:
        yield client


class TestHealth:
    """Health endpoint."""

    def test_health_ok(self, api_client: TestClient) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["experiment"] == "hipporag"


class TestCorpusAndRetrieve:
    """Corpus ingest, index, retrieve flow."""

    def test_index_and_retrieve_toy_flow(
        self,
        api_client: TestClient,
        toy_openie_by_passage: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from hipporag_core.openie import FixedOpenIEExtractor

        extractor = FixedOpenIEExtractor(toy_openie_by_passage)
        monkeypatch.setattr("server.routes.index.get_openie_extractor", lambda: extractor)
        monkeypatch.setattr("server.routes.retrieve.get_openie_extractor", lambda: extractor)

        for pid, extraction in toy_openie_by_passage.items():
            api_client.post(
                "/corpus/passages",
                json={"id": pid, "text": f"passage body for {pid}"},
            )

        index_resp = api_client.post("/index")
        assert index_resp.status_code == 200
        assert index_resp.json()["indexed_passages"] >= 1

        retrieve_resp = api_client.post(
            "/retrieve",
            json={
                "query": "Which Stanford professor works on Alzheimer's?",
                "top_k": 3,
            },
        )
        assert retrieve_resp.status_code == 200
        hits = retrieve_resp.json()["hits"]
        assert len(hits) >= 1

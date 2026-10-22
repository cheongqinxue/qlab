"""FastAPI routes (in-memory backend)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch):
    """App with memory graph store (no Neo4j)."""

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

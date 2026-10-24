"""FastAPI routes (in-memory backend)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _load_path_setup():
    """Load ``tests/_path_setup.py`` without importing as ``tests._path_setup``."""
    path = Path(__file__).resolve().parent / "_path_setup.py"
    spec = importlib.util.spec_from_file_location("hipporag_path_setup", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_hipporag_app():
    """Import HippoRAG ``main:app`` without colliding with other experiments."""
    path_setup = _load_path_setup()
    path_setup.ensure_hipporag_paths()
    path_setup.purge_shared_agent_modules()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("hipporag_main", root / "main.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.app


@pytest.fixture(autouse=True)
def hipporag_api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-memory graph and no Langfuse side effects (shared conftest stubs)."""
    monkeypatch.setenv("HIPPORAG_GRAPH_BACKEND", "memory")
    monkeypatch.setenv("HIPPORAG_BOOTSTRAP_TOY", "0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")


@pytest.fixture
def api_client() -> TestClient:
    """App with memory graph store (no Neo4j)."""
    with TestClient(_load_hipporag_app()) as client:
        yield client


class TestHealth:
    """Health endpoint."""

    def test_health_ok(self, api_client: TestClient) -> None:
        """Health returns experiment id and ok status."""
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
        """Ingest passages, index, and retrieve with fixture OpenIE."""
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

"""Pytest fixtures for MemGraphRAG (no Neo4j, no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import importlib.util

from agents_utils.testing.langfuse_stubs import install_langfuse_stubs_if_missing

_path_setup_path = Path(__file__).resolve().parent / "_path_setup.py"
_spec = importlib.util.spec_from_file_location("mgr_path_setup", _path_setup_path)
_mgr_path = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mgr_path)
ensure_memgraphrag_paths = _mgr_path.ensure_memgraphrag_paths
purge_shared_agent_modules = _mgr_path.purge_shared_agent_modules

ensure_memgraphrag_paths()

_MEMGRAPHRAG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = _MEMGRAPHRAG_ROOT / "config" / "default.json"


install_langfuse_stubs_if_missing()
purge_shared_agent_modules()

import pytest


@pytest.fixture(autouse=True)
def memgraphrag_test_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hash_encoder,
) -> None:
    """Point settings at repo config and avoid optional observability side effects."""
    cfg_data = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg_data["memory_data_dir"] = str(tmp_path / "memory")
    cfg_data["neo4j_enabled"] = False
    custom_config = tmp_path / "memgraphrag-config.json"
    custom_config.write_text(json.dumps(cfg_data), encoding="utf-8")
    monkeypatch.setenv("MEMGRAPHRAG_CONFIG", str(custom_config))
    monkeypatch.setenv("MEMGRAPHRAG_BOOTSTRAP_TOY", "0")
    monkeypatch.setenv("MEMGRAPHRAG_AGENTS", "fixture")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    purge_shared_agent_modules()
    from server.settings import get_settings

    get_settings.cache_clear()

    import memgraphrag_core.runtime as runtime_mod

    _orig_build_runtime = runtime_mod.build_runtime

    def _build_runtime(config, *, encoder=None):
        return _orig_build_runtime(config, encoder=encoder or hash_encoder)

    monkeypatch.setattr(runtime_mod, "build_runtime", _build_runtime)


@pytest.fixture
def hash_encoder():
    """Deterministic dense encoder for unit tests (no HF download)."""
    from agents_utils.memory.hash_dense_encoder import HashDenseEncoder

    return HashDenseEncoder()


@pytest.fixture
def three_layer_with_newton_facts():
    """Two conflicting facts in M_fac for agent tests."""
    from memgraphrag_core.memory.three_layer import ThreeLayerMemory
    from memgraphrag_core.models import FactRecord, PassageRecord

    memory = ThreeLayerMemory()
    memory.add_passage(PassageRecord(id="p1", text="Newton born 1642."))
    memory.add_passage(PassageRecord(id="p2", text="Newton born 1643."))
    memory.add_fact(
        FactRecord(id="f_newton_1642", text="Born 1642", passage_ids=["p1"])
    )
    memory.add_fact(
        FactRecord(id="f_newton_1643", text="Born 1643", passage_ids=["p2"])
    )
    return memory


@pytest.fixture
def toy_extraction_by_passage() -> dict[str, dict]:
    """Minimal extraction fixture keyed by passage id."""
    return {
        "p1": {
            "schemas": [{"id": "s_birth", "label": "PersonBirthYear"}],
            "facts": [{"id": "f1", "text": "Born 1642", "passage_ids": ["p1"]}],
        },
    }

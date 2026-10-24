"""Pytest fixtures for H-MEM (fixture agents, no LLM)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agents_utils.testing.langfuse_stubs import install_langfuse_stubs_if_missing

_path_setup_path = Path(__file__).resolve().parent / "_path_setup.py"
_spec = importlib.util.spec_from_file_location("hmem_path_setup", _path_setup_path)
_hmem_path = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hmem_path)
ensure_hmem_paths = _hmem_path.ensure_hmem_paths
purge_shared_agent_modules = _hmem_path.purge_shared_agent_modules

ensure_hmem_paths()
install_langfuse_stubs_if_missing()
purge_shared_agent_modules()


@pytest.fixture(autouse=True)
def hmem_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force fixture mode and disable optional observability."""
    hmem_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("HMEM_CONFIG", str(hmem_root / "config" / "default.json"))
    monkeypatch.setenv("HMEM_AGENTS", "fixture")
    monkeypatch.setenv("HMEM_BOOTSTRAP_TOY", "0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    purge_shared_agent_modules()
    from server.settings import get_settings

    get_settings.cache_clear()

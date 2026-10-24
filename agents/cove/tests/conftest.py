"""Pytest fixtures for CoVe (fixture agents only)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

_path_setup_path = Path(__file__).resolve().parent / "_path_setup.py"
_spec = importlib.util.spec_from_file_location("cove_path_setup", _path_setup_path)
_cove_path = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cove_path)
ensure_cove_paths = _cove_path.ensure_cove_paths
purge_shared_agent_modules = _cove_path.purge_shared_agent_modules

ensure_cove_paths()

_COVE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = _COVE_ROOT / "config" / "default.json"


def _install_langfuse_stubs() -> None:
    if "langfuse.langchain" in sys.modules:
        return
    langfuse_pkg = types.ModuleType("langfuse")
    langchain_pkg = types.ModuleType("langfuse.langchain")
    langchain_pkg.CallbackHandler = object
    langfuse_pkg.Langfuse = object
    sys.modules["langfuse"] = langfuse_pkg
    sys.modules["langfuse.langchain"] = langchain_pkg


_install_langfuse_stubs()
purge_shared_agent_modules()

import pytest


@pytest.fixture(autouse=True)
def cove_test_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point settings at temp config and force fixture agents."""
    cfg_data = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg_data["runs_data_dir"] = str(tmp_path / "runs")
    custom_config = tmp_path / "cove-config.json"
    custom_config.write_text(json.dumps(cfg_data), encoding="utf-8")
    monkeypatch.setenv("COVE_CONFIG", str(custom_config))
    monkeypatch.setenv("COVE_AGENTS", "fixture")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    purge_shared_agent_modules()
    from server.settings import get_settings

    get_settings.cache_clear()


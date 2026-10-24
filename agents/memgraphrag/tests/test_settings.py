"""Tests for server settings."""

from __future__ import annotations

from pathlib import Path

import pytest

import importlib.util
from pathlib import Path as _Path


def _purge() -> None:
    _path = _Path(__file__).resolve().parent / "_path_setup.py"
    _spec = importlib.util.spec_from_file_location("mgr_path_setup", _path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    _mod.purge_shared_agent_modules()


class TestGetSettings:
    """Environment-backed configuration."""

    def test_loads_from_default_config_file(self) -> None:
        _purge()
        from server.settings import Settings, get_settings

        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.retrieval_top_k >= 1
        assert 0.0 < settings.ppr_damping <= 1.0

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _purge()
        from server.settings import get_settings

        cfg = tmp_path / "custom.json"
        cfg.write_text('{"retrieval_top_k": 7, "ppr_damping": 0.25}', encoding="utf-8")
        monkeypatch.setenv("MEMGRAPHRAG_CONFIG", str(cfg))
        monkeypatch.setenv("MEMGRAPHRAG_AGENT_PORT", "9099")
        monkeypatch.setenv("MEMGRAPHRAG_TOP_K", "9")
        monkeypatch.setenv("MEMGRAPHRAG_PPR_DAMPING", "0.33")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.agent_port == 9099
        assert settings.config_path == cfg
        assert settings.retrieval_top_k == 9
        assert settings.ppr_damping == 0.33

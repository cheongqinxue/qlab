"""Tests for BEV store path resolution."""

from pathlib import Path

from bev.config import product_root, resolve_store_dir, static_dir


def test_resolve_store_dir_explicit(tmp_path: Path) -> None:
    custom = tmp_path / "my-bev-store"
    assert resolve_store_dir(custom) == custom.resolve()


def test_resolve_store_dir_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BEV_STORE_DIR", str(tmp_path / "from-env"))
    assert resolve_store_dir(None) == (tmp_path / "from-env").resolve()


def test_product_root_and_static_dir() -> None:
    root = product_root()
    assert root.name == "bev"
    assert (root / "webui").is_dir()
    assert static_dir() == root / "static"

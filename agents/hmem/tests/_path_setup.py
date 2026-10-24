"""Ensure H-MEM wins for shared module names (``main``, ``server``)."""

from __future__ import annotations

import sys
from pathlib import Path

_HMEM_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_ROOT = _HMEM_ROOT.parent


def ensure_hmem_paths() -> None:
    """Prepend H-MEM (and ``agents_utils`` parent) to ``sys.path``."""
    for path in (_AGENTS_ROOT, _HMEM_ROOT):
        entry = str(path)
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def purge_shared_agent_modules() -> None:
    """Remove cached ``main`` / ``server`` modules after reordering ``sys.path``."""
    ensure_hmem_paths()
    for name in list(sys.modules):
        if name == "main" or name.startswith("server"):
            del sys.modules[name]

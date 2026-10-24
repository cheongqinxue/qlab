"""Ensure HippoRAG wins for shared module names (``main``, ``server``)."""

from __future__ import annotations

import sys
from pathlib import Path

_HIPPORAG_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_ROOT = _HIPPORAG_ROOT.parent


def ensure_hipporag_paths() -> None:
    """Prepend HippoRAG (and ``agents_utils`` parent) to ``sys.path``.

    Raises no errors; mutates ``sys.path`` so ``server`` resolves to this harness.
    """
    for path in (_AGENTS_ROOT, _HIPPORAG_ROOT):
        entry = str(path)
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def purge_shared_agent_modules() -> None:
    """Remove cached ``main`` / ``server`` modules after reordering ``sys.path``."""
    ensure_hipporag_paths()
    for name in list(sys.modules):
        if name == "main" or name.startswith("server"):
            del sys.modules[name]

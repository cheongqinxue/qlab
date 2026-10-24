"""Ensure MemGraphRAG wins over HippoRAG for shared module names."""

from __future__ import annotations

import sys
from pathlib import Path

_MEMGRAPHRAG_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_ROOT = _MEMGRAPHRAG_ROOT.parent
_HIPPORAG_ROOT = _AGENTS_ROOT / "hipporag"


def ensure_memgraphrag_paths() -> None:
    """Prepend MemGraphRAG (and deps) to ``sys.path``."""
    for path in (_AGENTS_ROOT, _HIPPORAG_ROOT, _MEMGRAPHRAG_ROOT):
        entry = str(path)
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def purge_shared_agent_modules() -> None:
    """Remove cached ``main`` / ``server`` modules."""
    ensure_memgraphrag_paths()
    for name in list(sys.modules):
        if name == "main" or name.startswith("server"):
            del sys.modules[name]

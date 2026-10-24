"""Ensure CoVe wins over other experiments for shared module names."""

from __future__ import annotations

import sys
from pathlib import Path

_COVE_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_ROOT = _COVE_ROOT.parent


def ensure_cove_paths() -> None:
    """Prepend CoVe and agents tree to ``sys.path`` for harness-local imports."""
    for path in (_AGENTS_ROOT, _COVE_ROOT):
        entry = str(path)
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def purge_shared_agent_modules() -> None:
    """Remove cached ``main`` / ``server`` modules after reordering ``sys.path``."""
    ensure_cove_paths()
    for name in list(sys.modules):
        if name == "main" or name.startswith("server"):
            del sys.modules[name]

"""Mutable application state attached to FastAPI."""

from __future__ import annotations

from dataclasses import dataclass

from cove_core.runtime import CoveRuntime


@dataclass
class AppState:
    """Runtime handles for the CoVe server."""

    runtime: CoveRuntime

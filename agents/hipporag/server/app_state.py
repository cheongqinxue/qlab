"""Mutable application state attached to FastAPI."""

from __future__ import annotations

from dataclasses import dataclass

from hipporag_core.runtime import HipporagRuntime


@dataclass
class AppState:
    """Runtime handles for the HippoRAG server."""

    runtime: HipporagRuntime

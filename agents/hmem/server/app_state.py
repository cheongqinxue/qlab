"""Application state attached to FastAPI."""

from __future__ import annotations

from dataclasses import dataclass

from hmem_core.runtime import HMemRuntime


@dataclass
class AppState:
    """Runtime handles for the H-MEM service."""

    runtime: HMemRuntime

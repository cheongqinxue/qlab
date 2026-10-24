"""Minimal Langfuse modules for pytest when the SDK is not required."""

from __future__ import annotations

import sys
import types


def install_langfuse_stubs_if_missing() -> None:
    """Register stub ``langfuse`` packages so imports succeed without the SDK.

    The stub ``Langfuse`` client accepts arbitrary kwargs so
    ``ensure_langfuse_client()`` is safe when env keys are set in CI or local
    ``.env`` but experiments disable tracing in tests.
    """
    if "langfuse.langchain" in sys.modules:
        return

    class _LangfuseStub:
        def __init__(self, **kwargs: object) -> None:
            pass

    class _CallbackHandler:
        pass

    langfuse_pkg = types.ModuleType("langfuse")
    langchain_pkg = types.ModuleType("langfuse.langchain")
    langfuse_pkg.Langfuse = _LangfuseStub
    langchain_pkg.CallbackHandler = _CallbackHandler
    sys.modules["langfuse"] = langfuse_pkg
    sys.modules["langfuse.langchain"] = langchain_pkg

"""LangChain tool assembly for the ReHarness task agent."""

from __future__ import annotations

from typing import Any

from agents_utils.webfetch import create_tools as create_webfetch_tools
from agents_utils.websearch import create_tools as create_websearch_tools
from server.settings import Settings
from tools.python_runner import run_python


def build_agent_tools(settings: Settings) -> list[Any]:
    """Build tools bound into the task graph (python runner plus optional web tools).

    Args:
        settings: Server settings; when ``web_tools_enabled`` is false, only
            ``run_python`` is returned.

    Returns:
        LangChain-compatible tools for ``build_task_graph``.
    """
    tools: list[Any] = [run_python]
    if not settings.web_tools_enabled:
        return tools
    tools.extend(create_websearch_tools(settings))
    tools.extend(create_webfetch_tools(settings))
    return tools

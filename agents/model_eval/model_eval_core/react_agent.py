"""LangGraph prebuilt ReAct agent wrapper."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from model_eval_core.tools import default_eval_tools

REACT_SYSTEM_PROMPT = """You are a careful assistant in an evaluation harness.
Use tools when they help answer precisely. Follow user instructions exactly.
If you lack evidence, say you do not know instead of guessing.
When the user asks you to use a specific tool, use it.
"""


def build_react_agent(
    llm: BaseChatModel,
    *,
    tools: list[Any] | None = None,
    system_prompt: str = REACT_SYSTEM_PROMPT,
) -> Any:
    """Create a compiled LangGraph ReAct agent.

    Args:
        llm: Chat model (Ollama/OpenAI-compatible).
        tools: Optional tool override; defaults to :func:`default_eval_tools`.
        system_prompt: System message for the prebuilt agent.

    Returns:
        Compiled LangGraph agent graph.
    """
    tool_list = tools if tools is not None else default_eval_tools()
    return create_react_agent(
        llm,
        tool_list,
        prompt=system_prompt,
        name="model_eval_react",
    )

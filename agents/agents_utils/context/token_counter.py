"""Token counting for context pressure checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage

from .base_context import TokenCounter


def llm_token_counter(
    model: BaseLanguageModel,
    *,
    tools: Sequence[Any] | None = None,
) -> TokenCounter:
    """Build a counter using ``BaseLanguageModel.get_num_tokens_from_messages``.

    Provider chat models (e.g. ``ChatOpenAI``) implement this with model-specific
    tokenizers (tiktoken, etc.). Pass the same ``tools`` bound to the model when
    tool schemas affect the prompt.

    See LangChain ``BaseLanguageModel.get_num_tokens_from_messages`` and LangGraph
    short-term memory docs (LangMem), which recommend this over approximate counts.

    Args:
        model: Chat model used in the LangGraph agent.
        tools: Optional tool schemas passed through to the model counter.

    Returns:
        Callable suitable for ``AgentContextManager`` token counting.
    """

    def count(messages: Sequence[BaseMessage]) -> int:
        return model.get_num_tokens_from_messages(list(messages), tools=tools)

    return count

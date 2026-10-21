"""Serialize LangChain messages for harness prompts."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def messages_to_trajectory_text(messages: list[BaseMessage]) -> str:
    """Convert conversation messages to plain text for πM and πR.

    Args:
        messages: Conversation tail (no system message required).

    Returns:
        Human-readable trajectory string.
    """
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = f"tool:{message.name or 'unknown'}"
        else:
            role = message.type
        content = message.content
        if isinstance(content, list):
            text = str(content)
        else:
            text = str(content or "")
        if isinstance(message, AIMessage) and message.tool_calls:
            tools = ", ".join(tc["name"] for tc in message.tool_calls if tc.get("name"))
            lines.append(f"{role} (tools: {tools}): {text}")
        else:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)

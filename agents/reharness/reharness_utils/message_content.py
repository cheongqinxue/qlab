"""Extract user-visible text from LangChain / OpenAI Responses message content."""

from __future__ import annotations

import ast
from typing import Any


def extract_visible_assistant_text(content: Any) -> str:
    """Return only user-facing assistant text (no reasoning / encrypted blocks).

    OpenAI Responses API content is often a list of blocks (``reasoning``, ``text``).
    Prefer blocks with ``phase == final_answer`` when present.

    Args:
        content: ``AIMessage.content`` or interrupt prompt payload.

    Returns:
        Plain text suitable for chat UI and feedback prompts.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return content
            if isinstance(parsed, list):
                return extract_visible_assistant_text(parsed)
        return content
    if isinstance(content, list):
        text_blocks = [block for block in content if _is_text_block(block)]
        if not text_blocks:
            return ""
        final_blocks = [block for block in text_blocks if block.get("phase") == "final_answer"]
        chosen = final_blocks or text_blocks
        parts = [str(block.get("text", "")).strip() for block in chosen]
        return "\n\n".join(part for part in parts if part)
    return str(content)


def extract_reasoning_summary(content: Any) -> str | None:
    """Return a short reasoning summary when the model exposes one.

    Args:
        content: Message content (list of blocks or string).

    Returns:
        Summary text, or None when reasoning is hidden or absent.
    """
    blocks = _as_block_list(content)
    if blocks is None:
        return None
    summaries: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") not in ("reasoning", "thinking"):
            continue
        summary = block.get("summary")
        if isinstance(summary, list):
            summaries.extend(str(item).strip() for item in summary if str(item).strip())
        elif isinstance(summary, str) and summary.strip():
            summaries.append(summary.strip())
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            summaries.append(text.strip())
    joined = "\n".join(summaries).strip()
    return joined or None


def _is_text_block(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == "text"


def _as_block_list(content: Any) -> list[Any] | None:
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return None
            if isinstance(parsed, list):
                return parsed
    return None

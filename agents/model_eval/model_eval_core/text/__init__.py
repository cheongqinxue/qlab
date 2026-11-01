"""Text normalization for eval traces (visible answer vs chain-of-thought)."""


def strip_thinking(text: str) -> str:
    """Return user-visible text after optional chain-of-thought blocks."""
    think_close = "</" + "redacted_thinking" + ">"
    legacy_think_close = "</" + "think" + ">"
    for close_tag in (legacy_think_close, think_close):
        idx = text.lower().find(close_tag.lower())
        if idx != -1:
            return text[idx + len(close_tag) :].strip()
    return text.strip()


__all__ = ["strip_thinking"]

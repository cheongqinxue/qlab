"""Name pattern matching for hook and parameter selection."""

from __future__ import annotations

import fnmatch
import re


def name_matches(name: str, patterns: list[str]) -> bool:
    """Return True if ``name`` matches any fnmatch glob or ``re:`` regex."""
    if not patterns:
        return False
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.fullmatch(pattern[3:], name):
                return True
        elif fnmatch.fnmatch(name, pattern):
            return True
    return False


def list_matching_modules(model, patterns: list[str]) -> list[str]:
    """Return module names that match any pattern."""
    return [
        name
        for name, _ in model.named_modules()
        if name and name_matches(name, patterns)
    ]


def list_matching_parameters(model, patterns: list[str]) -> list[str]:
    """Return parameter names that match any pattern."""
    return [name for name, _ in model.named_parameters() if name_matches(name, patterns)]

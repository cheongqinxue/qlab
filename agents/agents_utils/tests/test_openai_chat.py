"""Tests for shared OpenAI chat factory."""

from __future__ import annotations

import pytest

from agents_utils.llm import openai_chat


class TestResolveOpenaiTemperature:
    """Environment-driven temperature resolution."""

    def test_unset_defaults_to_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Legacy harnesses expect deterministic temperature 0 when unset."""
        monkeypatch.delenv("OPENAI_TEMPERATURE", raising=False)
        assert openai_chat.resolve_openai_temperature() == 0.0

    def test_default_keyword_omits_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider-default models (e.g. gpt-6-luna) use OPENAI_TEMPERATURE=default."""
        monkeypatch.setenv("OPENAI_TEMPERATURE", "default")
        assert openai_chat.resolve_openai_temperature() is None

    def test_explicit_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Numeric string parses to float."""
        monkeypatch.setenv("OPENAI_TEMPERATURE", "0.2")
        assert openai_chat.resolve_openai_temperature() == 0.2

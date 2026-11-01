"""Ollama tool-binding error detection."""

from __future__ import annotations

from model_eval_core.ollama_tool_errors import (
    FAILURE_MODE_OLLAMA_TOOL_BINDING_UNSUPPORTED,
    is_ollama_tool_binding_unsupported,
    probe_tool_binding_unsupported,
)


class _FakeBadRequest(Exception):
    """Minimal OpenAI BadRequestError shape for unit tests."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        body: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class TestOllamaToolBindingUnsupported:
    """Match only documented Ollama tool/grammar refusal messages."""

    def test_detects_grammar_error(self) -> None:
        exc = _FakeBadRequest(
            message="Failed to initialize samplers: failed to parse grammar",
            body={"error": {"message": "failed to parse grammar"}},
        )
        assert is_ollama_tool_binding_unsupported(exc) is True

    def test_detects_does_not_support_tools(self) -> None:
        exc = _FakeBadRequest(message="model does not support tools")
        assert is_ollama_tool_binding_unsupported(exc) is True

    def test_rejects_unrelated_bad_request(self) -> None:
        exc = _FakeBadRequest(message="context length exceeded")
        assert is_ollama_tool_binding_unsupported(exc) is False

    def test_rejects_non_400_status(self) -> None:
        exc = _FakeBadRequest(message="failed to parse grammar", status_code=503)
        assert is_ollama_tool_binding_unsupported(exc) is False

    def test_probe_tags_failure_mode(self) -> None:
        exc = _FakeBadRequest(message="failed to parse grammar")
        probe = probe_tool_binding_unsupported(exc)
        assert probe.failure_modes == [FAILURE_MODE_OLLAMA_TOOL_BINDING_UNSUPPORTED]
        assert probe.signals["error_type"] == "_FakeBadRequest"

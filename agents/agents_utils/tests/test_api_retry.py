"""Tests for HTTP retry helpers."""

from __future__ import annotations

import pytest

from agents_utils.http.api_retry import (
    ApiClientError,
    ApiRetry,
    RetryableApiError,
    is_retryable_http_status,
)


class TestApiRetry:
    """Retry policy behavior."""

    def test_is_retryable_http_status(self) -> None:
        """429 and 503 are retried; 404 is not."""
        assert is_retryable_http_status(429) is True
        assert is_retryable_http_status(404) is False

    def test_retries_then_raises(self) -> None:
        """Exhausted retries re-raise RetryableApiError."""
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            raise RetryableApiError("transient")

        retry = ApiRetry(operation_name="test", max_attempts=2)
        with pytest.raises(RetryableApiError):
            retry.call(flaky)
        assert calls["n"] == 2

    def test_client_error_not_retried(self) -> None:
        """Permanent errors fail on first attempt."""
        def fail() -> None:
            raise ApiClientError("bad request")

        retry = ApiRetry(max_attempts=3)
        with pytest.raises(ApiClientError):
            retry.call(fail)

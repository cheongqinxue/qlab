"""Retry helpers for external HTTP and API calls (tenacity-backed)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class RetryableApiError(Exception):
    """Transient failure that may succeed on retry."""


class ApiClientError(Exception):
    """Non-retryable client or validation error."""


def is_retryable_http_status(status_code: int) -> bool:
    """Return whether an HTTP status code should be retried.

    Args:
        status_code: HTTP response status.

    Returns:
        True when the status is in :data:`RETRYABLE_HTTP_STATUS`.
    """
    return status_code in RETRYABLE_HTTP_STATUS


def classify_http_status(status_code: int, *, message: str) -> Exception:
    """Map an HTTP status to retryable or permanent errors.

    Args:
        status_code: HTTP response status.
        message: Error detail for the raised exception.

    Returns:
        :class:`RetryableApiError` or :class:`ApiClientError`.
    """
    if is_retryable_http_status(status_code):
        return RetryableApiError(message)
    return ApiClientError(message)


def http_status_error(exc: httpx.HTTPStatusError) -> Exception:
    """Convert ``httpx.HTTPStatusError`` to retryable or client errors.

    Args:
        exc: Raised HTTP status error from httpx.

    Returns:
        :class:`RetryableApiError` or :class:`ApiClientError`.
    """
    status = exc.response.status_code
    message = f"HTTP {status} for {exc.request.url}"
    return classify_http_status(status, message=message)


class ApiRetry:
    """Run a callable with bounded retries on transient errors."""

    def __init__(
        self,
        *,
        operation_name: str = "API call",
        max_attempts: int = 3,
    ) -> None:
        """Configure retry policy.

        Args:
            operation_name: Label for log messages.
            max_attempts: Maximum attempts including the first try.
        """
        self._operation_name = operation_name
        self._max_attempts = max(1, max_attempts)

    def call(self, fn: Callable[[], T]) -> T:
        """Invoke ``fn`` with retries on ``RetryableApiError`` only.

        Args:
            fn: Zero-argument callable to run.

        Returns:
            Result of ``fn`` on success.

        Raises:
            RetryableApiError: When all attempts are exhausted.
            ApiClientError: Propagated immediately without retry.
        """
        attempt = {"n": 0}

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type(RetryableApiError),
            before_sleep=lambda state: logger.warning(
                "Retrying %s (attempt %s/%s)",
                self._operation_name,
                state.attempt_number,
                self._max_attempts,
            ),
        )
        def _wrapped() -> T:
            attempt["n"] += 1
            return fn()

        try:
            return _wrapped()
        except RetryableApiError:
            logger.exception("%s failed after %s attempts", self._operation_name, self._max_attempts)
            raise


def create_api_retry(settings: Any, *, operation_name: str) -> ApiRetry:
    """Build ``ApiRetry`` from a settings object (optional ``api_retry_max_attempts``)."""
    max_attempts = int(getattr(settings, "api_retry_max_attempts", 3))
    return ApiRetry(operation_name=operation_name, max_attempts=max_attempts)

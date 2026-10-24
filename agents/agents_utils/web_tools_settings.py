"""Settings protocol for websearch / webfetch tool factories."""

from __future__ import annotations

from typing import Protocol


class WebToolsSettings(Protocol):
    """Minimal settings surface for ``agents_utils`` web tools.

    Experiment ``Settings`` dataclasses implement this protocol when optional
    ``websearch`` / ``webfetch`` tools are enabled (for example ReHarness with
    ``REHARNESS_WEB_TOOLS=1``).
    """

    websearch_provider: str
    langsearch_api_key: str
    webfetch_timeout_seconds: float
    webfetch_network_idle_timeout_ms: int
    webfetch_post_load_wait_ms: int
    webfetch_max_chars: int
    api_retry_max_attempts: int

"""Lazy OpenIE extractor singleton."""

from __future__ import annotations

from functools import lru_cache

from hipporag_core.openie import OpenIEExtractor, build_openie_extractor


@lru_cache
def get_openie_extractor() -> OpenIEExtractor:
    """Return shared OpenIE extractor."""
    return build_openie_extractor()

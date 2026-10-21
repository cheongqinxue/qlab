"""Regex catalog filtering behavior."""

from __future__ import annotations

from bev.store.catalog_query import filter_catalog_rows_by_regex, compile_metric_pattern


def test_filter_respects_limit_but_full_match_count() -> None:
    rows = [{"key": f"train/loss/{index}"} for index in range(100)]
    regex = compile_metric_pattern(r"train/loss")
    assert regex is not None
    matched, count, truncated = filter_catalog_rows_by_regex(rows, regex, limit=5)
    assert count == 100
    assert len(matched) == 5
    assert truncated is True

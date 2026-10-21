"""Tests for significant-figure formatting (mirrors web/src/lib/uplotTheme.js)."""

from __future__ import annotations


def format_significant_figures(value: float | None, significant_digits: int = 2) -> str:
    if value is None or not isinstance(value, (int, float)):
        return ""
    number = float(value)
    if number != number:  # NaN
        return ""
    return f"{number:.{significant_digits}g}"


class TestFormatSignificantFigures:
    def test_two_significant_figures(self) -> None:
        assert format_significant_figures(4.56) == "4.6"
        assert format_significant_figures(0.00123) == "0.0012"

    def test_empty_for_non_finite(self) -> None:
        assert format_significant_figures(None) == ""

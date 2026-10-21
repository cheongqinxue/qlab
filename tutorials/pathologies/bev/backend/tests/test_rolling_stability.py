"""Tests for causal rolling stability bands (mirrors web/src/lib/scale.js)."""

from __future__ import annotations

import math


def sample_std_dev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def rolling_stability_bands(
    points: list[tuple[float, float]],
    *,
    sigma: float = 2,
    min_history: int = 5,
    window: int | None = None,
) -> tuple[list[tuple[float, float | None]], list[tuple[float, float | None]]]:
    sorted_points = sorted(points, key=lambda pair: pair[0])
    upper: list[tuple[float, float | None]] = []
    lower: list[tuple[float, float | None]] = []
    history: list[float] = []

    for x, y in sorted_points:
        prior = history[-window:] if window else list(history)
        if len(prior) < min_history:
            upper.append((x, None))
            lower.append((x, None))
        else:
            mean = sum(prior) / len(prior)
            std_dev = sample_std_dev(prior) or 0.0
            upper.append((x, mean + sigma * std_dev))
            lower.append((x, mean - sigma * std_dev))
        history.append(y)

    return upper, lower


class TestRollingStabilityBands:
    def test_uses_only_prior_points(self) -> None:
        points = [(float(i), float(i)) for i in range(8)]
        upper, lower = rolling_stability_bands(points, min_history=3, sigma=2)
        assert upper[3][1] is not None
        assert upper[0][1] is None

    def test_constant_history_collapses_band(self) -> None:
        points = [(float(i), 1.0) for i in range(10)]
        upper, _lower = rolling_stability_bands(points, min_history=3, sigma=2)
        assert upper[-1][1] == 1.0

    def test_spike_outside_band(self) -> None:
        points = [(float(i), 1.0) for i in range(6)]
        points.append((6.0, 50.0))
        upper, lower = rolling_stability_bands(points, min_history=3, sigma=2)
        high = upper[-1][1]
        low = lower[-1][1]
        assert high is not None and low is not None
        assert 50.0 > high

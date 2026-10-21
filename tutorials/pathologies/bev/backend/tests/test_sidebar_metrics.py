"""Tests for sidebar metric-family panel choice (mirrors webui/src/lib/sidebarMetrics.js)."""


def panel_viz_for_family(plottable_count: int) -> str:
    return "line" if plottable_count == 1 else "multipanel"


def panel_partial_for_family(family_id: str, plottable_count: int) -> dict:
    scoped = family_id
    if plottable_count == 1:
        return {"metricKeys": [scoped], "title": scoped, "comparisonMetric": family_id}
    return {"comparisonMetric": family_id, "title": scoped}


def test_run_level_family_gets_line_panel() -> None:
    assert panel_viz_for_family(1) == "line"
    partial = panel_partial_for_family("train/loss", 1)
    assert partial["metricKeys"] == ["train/loss"]
    assert partial["comparisonMetric"] == "train/loss"


def test_architecture_family_gets_multipanel() -> None:
    assert panel_viz_for_family(2) == "multipanel"
    partial = panel_partial_for_family("train/grad_norm", 2)
    assert partial["comparisonMetric"] == "train/grad_norm"
    assert partial["title"] == "train/grad_norm"

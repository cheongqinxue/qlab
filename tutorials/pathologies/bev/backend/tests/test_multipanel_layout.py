"""Tests for multipanel grid layout (mirrors web/src/lib/multipanelLayout.js)."""

from __future__ import annotations


MULTIPANEL_MIN_COL_PX = 200
MULTIPANEL_PLOT_WIDTH_PX = MULTIPANEL_MIN_COL_PX
MULTIPANEL_CHART_GAP_PX = 2
MULTIPANEL_Y_AXIS_GUTTER_PX = 32


def multipanel_row_width_px(item_count: int) -> int:
    plot_w = MULTIPANEL_PLOT_WIDTH_PX
    g = MULTIPANEL_Y_AXIS_GUTTER_PX
    gap = MULTIPANEL_CHART_GAP_PX
    n = max(1, item_count)
    if n == 1:
        return plot_w + 2 * g
    return n * plot_w + 2 * g + (n - 1) * gap


def multipanel_cell_width_for_placement(placement: str) -> int:
    plot_w = MULTIPANEL_PLOT_WIDTH_PX
    g = MULTIPANEL_Y_AXIS_GUTTER_PX
    if placement == "both":
        return plot_w + 2 * g
    if placement in ("left", "right"):
        return plot_w + g
    return plot_w


def multipanel_columns_for_chart_area(chart_area_px: int, chart_count: int) -> int:
    capped = max(1, chart_count)
    for cols in range(capped, 0, -1):
        if multipanel_row_width_px(cols) <= chart_area_px:
            return cols
    return 1


def multipanel_grid_metrics(width_px: int, chart_count: int) -> dict[str, int]:
    if chart_count <= 0:
        return {"columns": 1, "plot_width_px": MULTIPANEL_PLOT_WIDTH_PX, "chart_area_px": 0}
    chart_area = max(0, width_px)
    if chart_area <= 0:
        return {"columns": 1, "plot_width_px": MULTIPANEL_PLOT_WIDTH_PX, "chart_area_px": 0}
    cols = multipanel_columns_for_chart_area(chart_area, chart_count)
    return {
        "columns": cols,
        "plot_width_px": MULTIPANEL_PLOT_WIDTH_PX,
        "chart_area_px": chart_area,
    }


def multipanel_columns_for_width(width_px: int, chart_count: int) -> int:
    return multipanel_grid_metrics(width_px, chart_count)["columns"]


def multipanel_chart_rows(keys: list[str], columns: int) -> list[list[str]]:
    cols = max(1, columns)
    rows: list[list[str]] = []
    for index in range(0, len(keys), cols):
        rows.append(keys[index : index + cols])
    return rows


def multipanel_cell_layout(index: int, total: int, columns: int) -> dict:
    cols = max(1, columns)
    rows = (total + cols - 1) // cols
    row = index // cols
    col = index % cols
    items_in_row = min(cols, total - row * cols)
    return {
        "row": row,
        "col": col,
        "is_first_row": row == 0,
        "is_first_col": col == 0,
        "is_last_col": col == items_in_row - 1,
        "is_last_row": row == rows - 1,
    }


def multipanel_y_axis_placement(cell: dict) -> str:
    if cell["is_first_col"] and cell["is_last_col"]:
        return "both"
    if cell["is_first_col"]:
        return "left"
    if cell["is_last_col"]:
        return "right"
    return "none"


def multipanel_cell_width_px(cell: dict) -> int:
    return multipanel_cell_width_for_placement(multipanel_y_axis_placement(cell))


def multipanel_frame_sides(_cell: dict) -> dict[str, bool]:
    return {
        "top": True,
        "left": True,
        "right": True,
        "bottom": True,
    }


class TestMultipanelColumnsForWidth:
    def test_caps_by_chart_count(self) -> None:
        assert multipanel_columns_for_width(1000, 2) == 2

    def test_respects_min_cell_width(self) -> None:
        assert multipanel_columns_for_width(500, 5) == 2

    def test_uses_full_host_width_for_chart_area(self) -> None:
        assert multipanel_columns_for_width(484, 10) == 2
        metrics = multipanel_grid_metrics(484, 10)
        assert metrics["chart_area_px"] == 484


class TestMultipanelGridMetrics:
    def test_full_row_width_includes_edge_gutters(self) -> None:
        n = 3
        assert multipanel_row_width_px(n) == (
            n * MULTIPANEL_PLOT_WIDTH_PX + 2 * MULTIPANEL_Y_AXIS_GUTTER_PX + (n - 1) * MULTIPANEL_CHART_GAP_PX
        )

    def test_edge_cells_wider_middle_matches_plot_target(self) -> None:
        assert multipanel_cell_width_for_placement("none") == MULTIPANEL_PLOT_WIDTH_PX
        assert multipanel_cell_width_for_placement("left") == MULTIPANEL_PLOT_WIDTH_PX + MULTIPANEL_Y_AXIS_GUTTER_PX
        assert multipanel_cell_width_for_placement("right") == MULTIPANEL_PLOT_WIDTH_PX + MULTIPANEL_Y_AXIS_GUTTER_PX
        assert multipanel_cell_width_for_placement("both") == MULTIPANEL_PLOT_WIDTH_PX + 2 * MULTIPANEL_Y_AXIS_GUTTER_PX

    def test_row_of_three_fits_chart_area(self) -> None:
        metrics = multipanel_grid_metrics(900, 9)
        n = 3
        row = sum(multipanel_cell_width_for_placement(p) for p in ("left", "none", "right"))
        row += (n - 1) * MULTIPANEL_CHART_GAP_PX
        assert row <= metrics["chart_area_px"]


class TestMultipanelChartRows:
    def test_splits_into_rows(self) -> None:
        assert multipanel_chart_rows(["a", "b", "c", "d"], 2) == [["a", "b"], ["c", "d"]]


class TestMultipanelYAxisPlacement:
    def test_wrapped_row_solo_chart_gets_both_axes(self) -> None:
        cell = multipanel_cell_layout(2, 3, 2)
        assert cell["is_first_col"] and cell["is_last_col"]
        assert multipanel_y_axis_placement(cell) == "both"

    def test_row_edges(self) -> None:
        left = multipanel_cell_layout(0, 4, 2)
        right = multipanel_cell_layout(1, 4, 2)
        assert multipanel_y_axis_placement(left) == "left"
        assert multipanel_y_axis_placement(right) == "right"


class TestMultipanelFrameSides:
    def test_each_cell_has_full_plot_bbox_frame(self) -> None:
        interior = multipanel_cell_layout(5, 12, 4)
        assert interior["is_first_row"] is False
        assert interior["is_first_col"] is False
        sides = multipanel_frame_sides(interior)
        assert sides == {"top": True, "right": True, "bottom": True, "left": True}

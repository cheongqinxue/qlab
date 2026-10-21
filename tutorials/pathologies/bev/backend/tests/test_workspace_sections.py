"""Tests for workspace section ordering (mirrors web/src/lib/workspace.js)."""

from __future__ import annotations


def move_section(sections: list[dict], section_id: str, delta: int) -> list[dict]:
    from_index = next((index for index, section in enumerate(sections) if section["id"] == section_id), -1)
    to_index = from_index + delta
    if from_index < 0 or to_index < 0 or to_index >= len(sections):
        return sections
    next_sections = [*sections]
    next_sections[from_index], next_sections[to_index] = (
        next_sections[to_index],
        next_sections[from_index],
    )
    return next_sections


def test_move_section_down() -> None:
    sections = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    moved = move_section(sections, "a", 1)
    assert [section["id"] for section in moved] == ["b", "a", "c"]


def test_move_section_up() -> None:
    sections = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    moved = move_section(sections, "c", -1)
    assert [section["id"] for section in moved] == ["a", "c", "b"]


def test_move_section_noop_at_bounds() -> None:
    sections = [{"id": "a"}, {"id": "b"}]
    assert move_section(sections, "a", -1) == sections
    assert move_section(sections, "b", 1) == sections


def reorder_section(sections: list[dict], dragged_id: str, target_id: str) -> list[dict]:
    from_index = next((index for index, section in enumerate(sections) if section["id"] == dragged_id), -1)
    to_index = next((index for index, section in enumerate(sections) if section["id"] == target_id), -1)
    if from_index < 0 or to_index < 0 or from_index == to_index:
        return sections
    next_sections = [*sections]
    item = next_sections.pop(from_index)
    next_sections.insert(to_index, item)
    return next_sections


def test_reorder_section_drag_drop() -> None:
    sections = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    moved = reorder_section(sections, "c", "a")
    assert [section["id"] for section in moved] == ["c", "a", "b"]

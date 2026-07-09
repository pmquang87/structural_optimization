"""find_last_anim: numeric (not lexicographic) ordering of animation indices."""
from __future__ import annotations

from oropt.runner import find_last_anim


def test_find_last_anim_numeric_past_a999(tmp_path):
    """OpenRadioss widens the anim index past A999 to A1000; lexicographically
    'A1000' < 'A999', so a plain sort silently extracted a MID-RUN state as
    the final one."""
    for i in (1, 500, 999, 1000, 1002):
        (tmp_path / f"mA{i:03d}").write_text("x", encoding="utf-8")
    last = find_last_anim(tmp_path, "m")
    assert last is not None and last.name == "mA1002"


def test_find_last_anim_normal_range(tmp_path):
    for i in (1, 2, 17):
        (tmp_path / f"mA{i:03d}").write_text("x", encoding="utf-8")
    assert find_last_anim(tmp_path, "m").name == "mA017"


def test_find_last_anim_empty(tmp_path):
    assert find_last_anim(tmp_path, "m") is None

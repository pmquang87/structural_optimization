"""The GUI's colour palette + validator (Streamlit-free, hermetic).

Guards that every curated dropdown colour is one pyvista actually accepts (so the
dropdown can never offer a value that fails at render time) and that the validator
mirrors pyvista's parser for the "Other…" hex/name escape hatch."""
from __future__ import annotations

import pytest

from oropt.gui.colors import COMMON_COLORS, OTHER, is_valid_color


def test_every_curated_colour_is_valid_and_unique():
    pytest.importorskip("pyvista")
    assert len(COMMON_COLORS) == len(set(COMMON_COLORS))     # no dupes in the menu
    bad = [c for c in COMMON_COLORS if not is_valid_color(c)]
    assert bad == [], f"curated colours pyvista rejects: {bad}"


def test_other_sentinel_is_not_a_real_colour():
    # The sentinel is a menu label, never forwarded to the renderer as a colour.
    assert OTHER not in COMMON_COLORS
    pytest.importorskip("pyvista")
    assert is_valid_color(OTHER) is False


def test_is_valid_color_accepts_hex_named_and_tab():
    pytest.importorskip("pyvista")
    for good in ("#b0c4de", "#ff8800", "mediumvioletred", "tab:blue", "GRAY"):
        assert is_valid_color(good) is True


def test_is_valid_color_rejects_typos_and_blank():
    pytest.importorskip("pyvista")
    assert is_valid_color("lightsteelbluu") is False         # the classic typo
    assert is_valid_color("#zzzzzz") is False
    assert is_valid_color("") is False
    assert is_valid_color("   ") is False
    assert is_valid_color(None) is False                     # type: ignore[arg-type]

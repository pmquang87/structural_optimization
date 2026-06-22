"""Colour choices for the GUI's animation pickers (Streamlit-free, unit-testable).

The evolution-animation surface/background colours are forwarded verbatim to
pyvista's ``add_mesh(color=...)`` / ``background_color`` in the off-screen render.
A free-text box lets a typo (``lightsteelbluu``) through to only fail later as a
non-zero render exit code. This module supplies a curated palette of common,
known-good names for a dropdown, plus :func:`is_valid_color` — pyvista's *own*
parser — so the GUI can validate an "Other…" hex/name entry in the form instead of
at render time.

Kept free of Streamlit (like :mod:`oropt.gui.views` / :mod:`oropt.gui.cases`) so it
imports fast and is hermetically testable; the Streamlit widget that uses it lives
in ``app.py``.
"""
from __future__ import annotations

# A curated subset of pyvista's ~213 named colours: the ones most useful for a part
# render, grouped roughly by hue. Every entry is a valid pyvista colour name; the
# picker adds an "Other…" escape hatch for any #hex code or other named colour.
COMMON_COLORS: tuple[str, ...] = (
    "lightsteelblue", "steelblue", "cornflowerblue", "royalblue", "dodgerblue",
    "navy", "cadetblue", "skyblue", "powderblue",
    "seagreen", "forestgreen", "mediumseagreen", "olivedrab", "lightgreen", "teal",
    "orange", "darkorange", "gold", "goldenrod", "tomato", "coral", "salmon",
    "firebrick", "crimson", "red", "brown", "sienna", "tan", "wheat",
    "purple", "indigo", "mediumpurple", "plum", "orchid",
    "white", "whitesmoke", "lightgray", "silver", "gray", "darkgray",
    "slategray", "dimgray", "black",
)

# Sentinel dropdown option that reveals the free-text hex/name box.
OTHER = "Other (hex / name)…"


def is_valid_color(value: str) -> bool:
    """Whether pyvista will accept *value* as a colour (named / ``#hex`` / ``tab:*``).

    Uses pyvista's own ``Color`` parser so it can't drift from what the renderer
    accepts. Blank is invalid. If pyvista can't be imported here, returns ``True``
    so the form never *blocks* on an import problem — a real bad colour would then
    just surface as a failed render, exactly as before.
    """
    value = (value or "").strip()
    if not value:
        return False
    try:
        import pyvista as pv
    except Exception:  # noqa: BLE001  (no pyvista here -> don't block the form)
        return True
    try:
        pv.Color(value)
        return True
    except Exception:  # noqa: BLE001
        return False

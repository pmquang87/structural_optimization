"""Pure helpers bridging the GUI's growth-box table (``st.data_editor`` rows) and
the :class:`~oropt.config.GrowthBox` config objects.

Kept Streamlit-free so the conversion is unit-testable and so importing it never
boots the Streamlit script. Empty data-editor cells arrive as ``None`` or float
``NaN`` (pandas blanks numeric columns); a row with any blank bound is dropped —
a partially-specified box is meaningless (and would otherwise silently default a
bound to 0.0 and select the wrong elements).
"""
from __future__ import annotations

from oropt.config import GrowthBox

# Columns of the growth-box data-editor, in display order.
BOX_COLUMNS = ["name", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
_BOUNDS = BOX_COLUMNS[1:]


def _is_blank(v) -> bool:
    """True for an empty editor cell: None, NaN, or whitespace-only text."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:        # NaN
        return True
    return isinstance(v, str) and v.strip() == ""


def records_from_growth_boxes(boxes) -> list[dict]:
    """Editor rows (one dict per box) from configured ``GrowthBox`` objects."""
    return [{"name": b.name, "x_min": b.x_min, "x_max": b.x_max,
             "y_min": b.y_min, "y_max": b.y_max,
             "z_min": b.z_min, "z_max": b.z_max} for b in boxes]


def growth_boxes_from_records(records) -> list[GrowthBox]:
    """``GrowthBox`` list from edited rows.

    Fully-empty rows are dropped, so the trailing blank row the dynamic editor
    offers never becomes a box; a row missing *any* bound is dropped too (all six
    bounds are required for a box to mean anything).
    """
    out: list[GrowthBox] = []
    for row in records:
        name = "" if _is_blank(row.get("name")) else str(row.get("name")).strip()
        bounds = [row.get(k) for k in _BOUNDS]
        if any(_is_blank(v) for v in bounds):
            continue
        out.append(GrowthBox(name=name,
                             **{k: float(v) for k, v in zip(_BOUNDS, bounds)}))
    return out

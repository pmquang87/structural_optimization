"""Pure helpers bridging the GUI's custom-camera-angle table (``st.data_editor``
rows) and the :class:`~oropt.config.CustomView` config objects.

Kept Streamlit-free so the conversion is unit-testable and so importing it never
boots the Streamlit script. A custom view is a built-in preset (``base``) plus
azimuth/elevation offsets, saved under a ``name`` the user can then pick as the
animation's camera angle. Empty editor cells arrive as ``None`` / float ``NaN``;
those fall back to sensible defaults (``base`` -> ``iso``, offsets -> ``0``).
"""
from __future__ import annotations

from oropt.config import CustomView

# Columns of the custom-view data-editor, in display order.
VIEW_COLUMNS = ["name", "base", "azimuth", "elevation"]


def _is_blank(v) -> bool:
    """True for an empty editor cell: None, NaN, or whitespace-only text."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:        # NaN
        return True
    return isinstance(v, str) and v.strip() == ""


def _num(v, default: float = 0.0) -> float:
    return default if _is_blank(v) else float(v)


def records_from_custom_views(custom_views) -> list[dict]:
    """Editor rows (one dict per angle) from configured ``CustomView`` objects."""
    return [{"name": cv.name, "base": cv.base, "azimuth": cv.azimuth,
             "elevation": cv.elevation} for cv in custom_views]


def custom_views_from_records(records) -> list[CustomView]:
    """``CustomView`` list from edited rows.

    Rows with no ``name`` are dropped (so the dynamic editor's trailing blank row
    never becomes an angle). A blank ``base`` defaults to ``iso``; blank offsets to
    ``0``. Names are kept as typed (the resolver matches them case-insensitively).
    """
    out: list[CustomView] = []
    for row in records:
        name = "" if _is_blank(row.get("name")) else str(row.get("name")).strip()
        if not name:
            continue
        base = ("iso" if _is_blank(row.get("base"))
                else str(row.get("base")).strip().lower())
        out.append(CustomView(
            name=name, base=base,
            azimuth=_num(row.get("azimuth")),
            elevation=_num(row.get("elevation"))))
    return out

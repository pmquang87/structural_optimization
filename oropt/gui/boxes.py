"""Pure helpers bridging the GUI's growth-region table (``st.data_editor`` rows)
and the :class:`~oropt.config.GrowthBox` config objects.

Kept Streamlit-free so the conversion is unit-testable and so importing it never
boots the Streamlit script. One table serves every shape via *nullable* columns:
a row carries ``name`` + ``shape`` and only the coordinate columns its shape uses
(``box`` -> six bounds, ``sphere`` -> centre + radius, ``cylinder`` -> two axis
end-points + radius); the columns the shape doesn't use are left blank. Empty
data-editor cells arrive as ``None`` or float ``NaN`` (pandas blanks numeric
columns); a row whose shape is missing any of *its* required coordinates is
dropped — a partially-specified region is meaningless (and would otherwise
silently default a coordinate to 0.0 and select the wrong elements).

The oriented-box local frame (origin / x_axis / xy_axis) and the ``deck_box_id``
reference are config/deck-authored (advanced) and deliberately not surfaced as
table columns, so they are preserved untouched on boxes that already carry them
only when a run reloads the YAML — the editor round-trips the coordinate shapes.
"""
from __future__ import annotations

from oropt.config import GrowthBox

# Coordinate fields required to fully specify each shape (all must be present for
# the row to become a box). ``name``/``shape`` are handled separately.
_REQUIRED: dict[str, list[str]] = {
    "box": ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
    "sphere": ["cx", "cy", "cz", "radius"],
    "cylinder": ["x1", "y1", "z1", "x2", "y2", "z2", "radius"],
}

# Every numeric column the editor shows (union across shapes), in display order.
_NUMERIC: list[str] = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
                       "cx", "cy", "cz", "radius",
                       "x1", "y1", "z1", "x2", "y2", "z2"]

# Full column order of the data-editor.
BOX_COLUMNS: list[str] = ["name", "shape", *_NUMERIC]


def _is_blank(v) -> bool:
    """True for an empty editor cell: None, NaN, or whitespace-only text."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:        # NaN
        return True
    return isinstance(v, str) and v.strip() == ""


def _shape_of(b: GrowthBox) -> str:
    kind = b.shape_kind()
    return kind if kind in _REQUIRED else "box"


def records_from_growth_boxes(boxes) -> list[dict]:
    """Editor rows (one dict per region) from configured ``GrowthBox`` objects.

    Each row carries ``name`` + ``shape`` and, for the columns the shape uses,
    that box's value; the columns another shape would use are left ``None`` so the
    editor shows them blank."""
    out: list[dict] = []
    for b in boxes:
        kind = _shape_of(b)
        req = set(_REQUIRED[kind])
        row: dict = {"name": b.name, "shape": kind}
        for col in _NUMERIC:
            row[col] = getattr(b, col) if col in req else None
        out.append(row)
    return out


def growth_boxes_from_records(records) -> list[GrowthBox]:
    """``GrowthBox`` list from edited rows.

    Fully-empty rows are dropped, so the trailing blank row the dynamic editor
    offers never becomes a region; a row missing *any* coordinate its shape needs
    is dropped too. An unrecognised shape is dropped (validation surfaces the typo
    on the coordinates path); a blank shape defaults to ``box``."""
    out: list[GrowthBox] = []
    for row in records:
        name = "" if _is_blank(row.get("name")) else str(row.get("name")).strip()
        raw_shape = row.get("shape")
        kind = "box" if _is_blank(raw_shape) else str(raw_shape).strip().lower()
        req = _REQUIRED.get(kind)
        if req is None:
            continue
        vals = [row.get(k) for k in req]
        if any(_is_blank(v) for v in vals):
            continue
        out.append(GrowthBox(name=name, shape=kind,
                             **{k: float(v) for k, v in zip(req, vals)}))
    return out

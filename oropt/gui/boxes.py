"""Pure helpers bridging the GUI's growth-region tables (``st.data_editor`` rows)
and the :class:`~oropt.config.GrowthBox` config objects.

Kept Streamlit-free so the conversion is unit-testable and so importing it never
boots the Streamlit script.

The **main region table** serves every shape via *nullable* columns: a row carries
``name`` + ``shape`` + an optional ``deck_box_id`` and only the coordinate columns
its shape uses (``box`` -> six bounds, ``sphere`` -> centre + radius, ``cylinder``
-> two axis end-points + radius); the columns the shape doesn't use are left blank.
Empty data-editor cells arrive as ``None`` or float ``NaN`` (pandas blanks numeric
columns); a row whose shape is missing any of *its* required coordinates is
dropped — a partially-specified region is meaningless (and would otherwise silently
default a coordinate to 0.0 and select the wrong elements). A row that names a
``deck_box_id`` needs no coordinates (they are read from the deck's ``/BOX`` card at
run start), so it is kept regardless.

The **oriented-box frame** (origin / local +x axis / in-plane vector) is edited in
a separate, narrower table keyed by region name (:func:`records_from_frames` /
:func:`apply_frame_records`) because a 3-vector doesn't fit a single numeric
column; a frame only applies to a ``box`` shape.
"""
from __future__ import annotations

import dataclasses

from oropt.config import GrowthBox

# Coordinate fields required to fully specify each shape (all must be present for
# the row to become a region). ``name``/``shape``/``deck_box_id`` are handled apart.
_REQUIRED: dict[str, list[str]] = {
    "box": ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
    "sphere": ["cx", "cy", "cz", "radius"],
    "cylinder": ["x1", "y1", "z1", "x2", "y2", "z2", "radius"],
}

# Every numeric column the main table shows (union across shapes), in display order.
_NUMERIC: list[str] = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
                       "cx", "cy", "cz", "radius",
                       "x1", "y1", "z1", "x2", "y2", "z2"]

# Full column order of the main region data-editor.
BOX_COLUMNS: list[str] = ["name", "shape", "deck_box_id", *_NUMERIC]

# Columns of the oriented-frame data-editor: name + origin + local +x + in-plane.
FRAME_COLUMNS: list[str] = ["name", "ox", "oy", "oz",
                            "ax", "ay", "az", "bx", "by", "bz"]


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


def _vec3(row, keys):
    """``[x, y, z]`` from three row cells, or ``None`` if any is blank."""
    vals = [row.get(k) for k in keys]
    if any(_is_blank(v) for v in vals):
        return None
    return [float(v) for v in vals]


# --------------------------------------------------------------------------- #
# main region table (name / shape / deck_box_id / per-shape coordinates)
# --------------------------------------------------------------------------- #
def records_from_growth_boxes(boxes) -> list[dict]:
    """Editor rows (one dict per region) from configured ``GrowthBox`` objects.

    Each row carries ``name`` + ``shape`` + ``deck_box_id`` and, for the columns
    the shape uses, that region's value; columns another shape would use — and all
    coordinates of a ``deck_box_id`` region (they come from the deck) — are left
    ``None`` so the editor shows them blank."""
    out: list[dict] = []
    for b in boxes:
        kind = _shape_of(b)
        req = set(_REQUIRED[kind])
        deck_ref = b.deck_box_id is not None
        row: dict = {"name": b.name, "shape": kind, "deck_box_id": b.deck_box_id}
        for col in _NUMERIC:
            row[col] = getattr(b, col) if (col in req and not deck_ref) else None
        out.append(row)
    return out


def growth_boxes_from_records(records) -> list[GrowthBox]:
    """``GrowthBox`` list from edited rows.

    Fully-empty rows are dropped, so the trailing blank row the dynamic editor
    offers never becomes a region; a row missing *any* coordinate its shape needs
    is dropped too — unless it names a ``deck_box_id``, whose coordinates come from
    the deck at run start. An unrecognised shape is dropped (validation surfaces the
    typo on the coordinates path); a blank shape defaults to ``box``. The oriented
    frame is applied separately (:func:`apply_frame_records`)."""
    out: list[GrowthBox] = []
    for row in records:
        name = "" if _is_blank(row.get("name")) else str(row.get("name")).strip()
        raw_shape = row.get("shape")
        kind = "box" if _is_blank(raw_shape) else str(raw_shape).strip().lower()
        req = _REQUIRED.get(kind)
        if req is None:
            continue
        deck_id = row.get("deck_box_id")
        if not _is_blank(deck_id):
            out.append(GrowthBox(name=name, shape=kind, deck_box_id=int(deck_id)))
            continue
        vals = [row.get(k) for k in req]
        if any(_is_blank(v) for v in vals):
            continue
        out.append(GrowthBox(name=name, shape=kind,
                             **{k: float(v) for k, v in zip(req, vals)}))
    return out


# --------------------------------------------------------------------------- #
# oriented-frame table (name -> origin / local +x axis / in-plane vector)
# --------------------------------------------------------------------------- #
def records_from_frames(boxes) -> list[dict]:
    """Frame-editor rows: one per ``box``-shaped region (a local frame only applies
    to a box), carrying its name and the frame components (origin ``ox/oy/oz``,
    local +x ``ax/ay/az``, in-plane vector ``bx/by/bz``), blank when it has none."""
    out: list[dict] = []
    for b in boxes:
        if b.shape_kind() != "box":
            continue
        o = b.origin or [None, None, None]
        a = b.x_axis or [None, None, None]
        c = b.xy_axis or [None, None, None]
        out.append({"name": b.name,
                    "ox": o[0], "oy": o[1], "oz": o[2],
                    "ax": a[0], "ay": a[1], "az": a[2],
                    "bx": c[0], "by": c[1], "bz": c[2]})
    return out


def apply_frame_records(boxes, records) -> list[GrowthBox]:
    """Return *boxes* with each ``box``-shaped region's oriented frame set from the
    matching-by-name frame row: the local +x axis (``ax/ay/az``), the in-plane
    vector (``bx/by/bz``) and an optional origin (``ox/oy/oz``, blank -> world
    origin). An all-blank row clears the frame. Non-box regions, and boxes with no
    matching row, are returned unchanged."""
    frames: dict[str, tuple] = {}
    for row in records:
        name = "" if _is_blank(row.get("name")) else str(row.get("name")).strip()
        frames[name] = (_vec3(row, ("ox", "oy", "oz")),
                        _vec3(row, ("ax", "ay", "az")),
                        _vec3(row, ("bx", "by", "bz")))
    out: list[GrowthBox] = []
    for b in boxes:
        name = b.name or ""
        if b.shape_kind() == "box" and name in frames:
            origin, x_axis, xy_axis = frames[name]
            b = dataclasses.replace(b, origin=origin, x_axis=x_axis,
                                    xy_axis=xy_axis)
        out.append(b)
    return out

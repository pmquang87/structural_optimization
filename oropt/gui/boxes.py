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

A **polyhedron**'s node list doesn't fit the fixed-column main table either
(N points of 3 coordinates each), so it is edited in its own name-keyed table —
one ``x``/``y``/``z`` row per node, dynamic rows (:func:`records_from_points` /
:func:`apply_point_records`). Its main-table row carries only ``name`` +
``shape``; every coordinate must be given explicitly in the points table (a row
missing any coordinate is dropped — no defaults, no inference).
"""
from __future__ import annotations

import dataclasses

from oropt.config import GrowthBox

# Coordinate fields required to fully specify each shape (all must be present for
# the row to become a region). ``name``/``shape``/``deck_box_id`` are handled
# apart. A polyhedron needs no main-table coordinates: its node list lives in the
# separate points table.
_REQUIRED: dict[str, list[str]] = {
    "box": ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
    "sphere": ["cx", "cy", "cz", "radius"],
    "cylinder": ["x1", "y1", "z1", "x2", "y2", "z2", "radius"],
    "polyhedron": [],
}

# Every numeric column the main table shows (union across shapes), in display order.
_NUMERIC: list[str] = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
                       "cx", "cy", "cz", "radius",
                       "x1", "y1", "z1", "x2", "y2", "z2"]

# Full column order of the main region data-editor. ``carve`` is the overlap
# policy checkbox: on (default) = a region overlapping the original part voids
# those elements too (carve-and-regrow); off = the original part stays alive and
# only expansion elements (ids > model.growth_original_elem_max) start void.
BOX_COLUMNS: list[str] = ["name", "shape", "carve", "deck_box_id", *_NUMERIC]

# Columns of the oriented-frame data-editor: name + origin + local +x + in-plane.
FRAME_COLUMNS: list[str] = ["name", "ox", "oy", "oz",
                            "ax", "ay", "az", "bx", "by", "bz"]

# Columns of the polyhedron-points data-editor: region name + one node per row.
POINT_COLUMNS: list[str] = ["name", "x", "y", "z"]


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

    Each row carries ``name`` + ``shape`` + ``carve`` + ``deck_box_id`` and, for
    the columns the shape uses, that region's value; columns another shape would
    use — and all coordinates of a ``deck_box_id`` region (they come from the
    deck) — are left ``None`` so the editor shows them blank."""
    out: list[dict] = []
    for b in boxes:
        kind = _shape_of(b)
        req = set(_REQUIRED[kind])
        deck_ref = b.deck_box_id is not None
        row: dict = {"name": b.name, "shape": kind, "carve": bool(b.carve),
                     "deck_box_id": b.deck_box_id}
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
    typo on the coordinates path); a blank shape defaults to ``box``; a blank
    ``carve`` cell defaults to on (carve-and-regrow, the historical behaviour).
    The oriented frame is applied separately (:func:`apply_frame_records`)."""
    out: list[GrowthBox] = []
    for row in records:
        name = "" if _is_blank(row.get("name")) else str(row.get("name")).strip()
        raw_shape = row.get("shape")
        kind = "box" if _is_blank(raw_shape) else str(raw_shape).strip().lower()
        req = _REQUIRED.get(kind)
        if req is None:
            continue
        raw_carve = row.get("carve")
        carve = True if _is_blank(raw_carve) else bool(raw_carve)
        deck_id = row.get("deck_box_id")
        if not _is_blank(deck_id):
            out.append(GrowthBox(name=name, shape=kind, carve=carve,
                                 deck_box_id=int(deck_id)))
            continue
        vals = [row.get(k) for k in req]
        if any(_is_blank(v) for v in vals):
            continue
        out.append(GrowthBox(name=name, shape=kind, carve=carve,
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


# --------------------------------------------------------------------------- #
# polyhedron-points table (name -> one x/y/z row per node)
# --------------------------------------------------------------------------- #
def records_from_points(boxes) -> list[dict]:
    """Points-editor rows: one per node of each polyhedron-shaped region,
    carrying the region's name and that node's ``x``/``y``/``z``. A polyhedron
    with no points yet gets a single blank row so its name shows up in the
    table ready to fill in. Other shapes contribute no rows."""
    out: list[dict] = []
    for b in boxes:
        if b.shape_kind() != "polyhedron":
            continue
        for p in (b.points or []):
            out.append({"name": b.name, "x": p[0], "y": p[1], "z": p[2]})
        if not b.points:
            out.append({"name": b.name, "x": None, "y": None, "z": None})
    return out


def apply_point_records(boxes, records) -> list[GrowthBox]:
    """Return *boxes* with each polyhedron-shaped region's points set from the
    rows matching its name (in row order): one node per row, all of ``x``/``y``/
    ``z`` required — a row missing any coordinate is dropped, never defaulted. A
    region whose name appears in no row is returned unchanged; a region whose
    rows are all incomplete gets ``points=None`` (deleting a region's node rows
    clears its points). Non-polyhedron regions, and the editor's fully-blank
    trailing row, are ignored."""
    pts: dict[str, list] = {}
    for row in records:
        blank_name = _is_blank(row.get("name"))
        v = _vec3(row, ("x", "y", "z"))
        if blank_name and v is None:
            continue                    # the dynamic editor's trailing blank row
        name = "" if blank_name else str(row.get("name")).strip()
        pts.setdefault(name, [])
        if v is not None:
            pts[name].append(v)
    out: list[GrowthBox] = []
    for b in boxes:
        name = b.name or ""
        if b.shape_kind() == "polyhedron" and name in pts:
            b = dataclasses.replace(b, points=pts[name] or None)
        out.append(b)
    return out

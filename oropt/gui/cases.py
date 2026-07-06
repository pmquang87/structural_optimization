"""Pure helpers bridging the GUI's load-case table (``st.data_editor`` rows) and
the :class:`~oropt.config.LoadCase` config objects.

Kept Streamlit-free so the conversion is unit-testable and so importing it never
boots the Streamlit script. Empty data-editor cells arrive as ``None`` or float
``NaN`` (pandas blanks numeric columns); those map back to ``None`` so the loop
inherits the model/constraints defaults.

A load case may constrain the displacement at **several** nodes, each with its
own limit, so those live in a single free-text column formatted
``node:limit; node:limit`` (e.g. ``10021367:1.0; 10021400:2.0``). A bare
``node`` (no ``:limit``) tracks the node but leaves it unconstrained. The column
round-trips :attr:`~oropt.config.LoadCase.disp_constraints`.
"""
from __future__ import annotations

from oropt.config import DispConstraint, LoadCase

# Columns of the load-case data-editor, in display order.
CASE_COLUMNS = ["name", "stem", "weight", "disp_constraints", "sigma_allow",
                "fast_mode"]


def _is_blank(v) -> bool:
    """True for an empty editor cell: None, NaN, or whitespace-only text."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:        # NaN
        return True
    return isinstance(v, str) and v.strip() == ""


def _opt_float(v):
    return None if _is_blank(v) else float(v)


def _as_bool(v) -> bool:
    """Checkbox cell -> bool. A blank cell (unchecked / the dynamic editor's new
    row) is False; text is read leniently so a hand-edited YAML/CSV round-trips."""
    if _is_blank(v):
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _fmt_number(v) -> str:
    """Compact number for the editor cell: an int-valued float shows no decimals."""
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:g}"


def format_disp_constraints(dcs) -> str:
    """``node:limit; node:limit`` text from a list of ``DispConstraint``.

    A blank limit renders as a bare node id (``node``), so the string round-trips
    an unconstrained-but-tracked node. Entries with no node id are skipped.
    """
    parts = []
    for dc in dcs:
        if dc.node_id is None:
            continue
        if dc.d_allow is None:
            parts.append(_fmt_number(dc.node_id))
        else:
            parts.append(f"{_fmt_number(dc.node_id)}:{_fmt_number(dc.d_allow)}")
    return "; ".join(parts)


def parse_disp_constraints(text) -> list[DispConstraint]:
    """List of ``DispConstraint`` from a ``node:limit; node:limit`` string.

    Tolerant of the free-text column: entries are separated by ``;`` (``,`` and
    newlines too), the node and limit by ``:``. A bare ``node`` -> unconstrained.
    Unparseable tokens (non-numeric node) are skipped rather than crashing the
    editor.
    """
    out: list[DispConstraint] = []
    if _is_blank(text):
        return out
    for tok in str(text).replace(",", ";").replace("\n", ";").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        node_s, _, lim_s = tok.partition(":")
        node_s, lim_s = node_s.strip(), lim_s.strip()
        if not node_s:
            continue
        try:
            node_id = int(float(node_s))
            d_allow = None if not lim_s else float(lim_s)
        except ValueError:
            continue
        out.append(DispConstraint(node_id=node_id, d_allow=d_allow))
    return out


def records_from_load_cases(load_cases) -> list[dict]:
    """Editor rows (one dict per case) from configured ``LoadCase`` objects."""
    return [{"name": lc.name, "stem": lc.stem, "weight": lc.weight,
             "disp_constraints": format_disp_constraints(lc.disp_constraints),
             "sigma_allow": lc.sigma_allow,
             "fast_mode": bool(lc.fast_mode)} for lc in load_cases]


def load_cases_from_records(records) -> list[LoadCase]:
    """``LoadCase`` list from edited rows.

    Fully-empty rows (no name *and* no stem) are dropped, so the trailing blank
    row the dynamic editor offers never becomes a case. Blank optional cells stay
    ``None`` so the loop inherits the model/constraints defaults; a blank weight
    defaults to 1.0 (an explicit 0 is preserved). The ``disp_constraints`` text is
    parsed into a per-node list.
    """
    out: list[LoadCase] = []
    for row in records:
        name = "" if _is_blank(row.get("name")) else str(row.get("name")).strip()
        stem = "" if _is_blank(row.get("stem")) else str(row.get("stem")).strip()
        if not name and not stem:
            continue
        w = _opt_float(row.get("weight"))
        out.append(LoadCase(
            name=name or "case",
            stem=stem,
            weight=1.0 if w is None else w,
            disp_constraints=parse_disp_constraints(row.get("disp_constraints")),
            sigma_allow=_opt_float(row.get("sigma_allow")),
            fast_mode=_as_bool(row.get("fast_mode"))))
    return out

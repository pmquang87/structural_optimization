"""Pure helpers bridging the GUI's load-case table (``st.data_editor`` rows) and
the :class:`~oropt.config.LoadCase` config objects.

Kept Streamlit-free so the conversion is unit-testable and so importing it never
boots the Streamlit script. Empty data-editor cells arrive as ``None`` or float
``NaN`` (pandas blanks numeric columns); those map back to ``None`` so the loop
inherits the model/constraints defaults.
"""
from __future__ import annotations

from oropt.config import LoadCase

# Columns of the load-case data-editor, in display order.
CASE_COLUMNS = ["name", "stem", "weight", "disp_node_id", "sigma_allow", "d_allow"]


def _is_blank(v) -> bool:
    """True for an empty editor cell: None, NaN, or whitespace-only text."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:        # NaN
        return True
    return isinstance(v, str) and v.strip() == ""


def _opt_int(v):
    return None if _is_blank(v) else int(float(v))


def _opt_float(v):
    return None if _is_blank(v) else float(v)


def records_from_load_cases(load_cases) -> list[dict]:
    """Editor rows (one dict per case) from configured ``LoadCase`` objects."""
    return [{"name": lc.name, "stem": lc.stem, "weight": lc.weight,
             "disp_node_id": lc.disp_node_id, "sigma_allow": lc.sigma_allow,
             "d_allow": lc.d_allow} for lc in load_cases]


def load_cases_from_records(records) -> list[LoadCase]:
    """``LoadCase`` list from edited rows.

    Fully-empty rows (no name *and* no stem) are dropped, so the trailing blank
    row the dynamic editor offers never becomes a case. Blank optional cells stay
    ``None`` so the loop inherits the model/constraints defaults; a blank weight
    defaults to 1.0 (an explicit 0 is preserved).
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
            disp_node_id=_opt_int(row.get("disp_node_id")),
            sigma_allow=_opt_float(row.get("sigma_allow")),
            d_allow=_opt_float(row.get("d_allow"))))
    return out

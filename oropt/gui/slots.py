"""Streamlit-free helpers bridging the GUI's per-slot CPU table (``st.data_editor``
rows) and the :attr:`~oropt.config.RunOpts.solver_slots` config (a list of
:class:`~oropt.config.SolverSlot`).

Kept Streamlit-free so the conversion is unit-testable and so importing it never
boots the Streamlit script — the same pattern as :mod:`oropt.gui.boxes` /
:mod:`oropt.gui.cases`.
"""
from __future__ import annotations

from oropt.config import SolverSlot

# Columns of the per-slot CPU data-editor: MPI domains x OpenMP threads.
SLOT_COLUMNS: list[str] = ["np", "nt"]


def _blank(v) -> bool:
    """True for an empty numeric editor cell: ``None`` or float ``NaN``."""
    return v is None or (isinstance(v, float) and v != v)


def records_from_slots(slots) -> list[dict]:
    """Editor rows (one dict per slot) from configured ``SolverSlot`` objects."""
    return [{"np": int(s.np), "nt": int(s.nt)} for s in (slots or [])]


def slots_from_records(records) -> list[SolverSlot]:
    """``SolverSlot`` list from edited rows. A fully-blank row (the dynamic
    editor's trailing row) is dropped; a partially-filled row defaults the missing
    field (``np=1`` / ``nt=12``), so a row is a slot as soon as either cell is
    filled."""
    out: list[SolverSlot] = []
    for row in records or []:
        np_v, nt_v = row.get("np"), row.get("nt")
        if _blank(np_v) and _blank(nt_v):
            continue
        out.append(SolverSlot(np=1 if _blank(np_v) else int(np_v),
                              nt=12 if _blank(nt_v) else int(nt_v)))
    return out

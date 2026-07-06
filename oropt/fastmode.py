"""Fast mode: swap a load case's full nonlinear solve for a validated TIED
LINEAR one (~35x faster) that still reads peak stress from the engine animation.

**Why the tie is needed.** On a converted contact deck the applied load *and* the
support are both contact-mediated (the loading cylinder floats ~0.3 mm off the
bore joined only by a soft bootstrap spring; the fixed support holds the part
through open TYPE7 contacts). A naive ``/IMPL/LINEAR`` step from t=0 therefore has
no load *and* no support path — the load rides the 100 N/mm spring (|u| ~ 40 mm,
0 design stress) and the part is an unsupported mechanism (indefinite K). The fix,
proven on the elevator-linkage pull case (FASTMODE_REPORT.md), is to bond both
contact patches so the linear stiffness matrix has a real load+support path:

  * **load side** — extend the loading rigid body's node group with the nearest
    design node of every one of its nodes (a rigid pin-in-hole load introduction);
  * **support side** — ground the design nodes sitting against the fixed support's
    contact-master surfaces by folding them into the fully-fixed rigid body.

The engine then reproduces the nonlinear peak von-Mises to within ~14% in ~3 min
vs ~1h49m. It is a **ranking/flagging screen, not a certifying stress** — the
tied linear read is ~14% non-conservative, which the calibrated
:data:`~oropt.config.FAST_MODE_SIGMA_ALLOW` (254 MPa) folds back in.

**Discovery, not hard-coding.** The tie topology is discovered from the deck
(:func:`discover_tie`): the loading rigid body is the one whose master node carries
the ``/CLOAD``; the fixed rigid body is the one whose master node is fully fixed
(``/BCS`` ``111 111``); the support contact-master surfaces are the ``/INTER/TYPE7``
masters that are ``/SURF/GRSHEL`` surfaces owned by the fixed rigid body. Ported
from ``make_tied2.py`` / ``map_contacts.py`` / ``find_loadpath.py`` in the
fastmode_test study.

The optimiser reads stress straight from the engine anim via
:func:`oropt.results.extract` (an offline tet4-CST recovery was unreliable on the
growth mesh); this module only builds the tied starter + a plain ``/IMPL/LINEAR``
engine deck (no ``/IMPL/PRINT/STIF`` — that was only for the offline matrix study).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .deck import Deck

# Design-node distance (mm) within which a node counts as seated against the
# fixed support surface and is grounded. The elevator-linkage support contacts had
# Gapmin 0.069 / 0.129 mm, so 0.30 mm captures the seated patch (make_tied2 THRESH).
DEFAULT_SUPPORT_THRESH = 0.30


class FastModeError(RuntimeError):
    """Raised when the tied-linear tie topology cannot be discovered from a deck.

    Fast mode only makes sense on a contact-mediated deck with a loading rigid
    body (its master node carries the ``/CLOAD``), a fully-fixed rigid body and
    fixed-support ``/INTER/TYPE7`` contacts. When any of those cannot be found the
    load/support tie is undefined, so we fail loudly at run start rather than
    silently solving a mechanism.
    """


# --------------------------------------------------------------------------- parse
@dataclass
class DeckIndex:
    """Load-path topology parsed from a full starter deck in one pass.

    Covers every ``/NODE`` block (not just the design part's — the rigid parts'
    nodes live in their own blocks), so tie coordinates resolve across the whole
    model. See :func:`index_deck`.
    """
    coords: dict = field(default_factory=dict)        # node id -> (x, y, z), all blocks
    grnod: dict = field(default_factory=dict)         # /GRNOD/NODE/<id> -> [node ids]
    grshel: dict = field(default_factory=dict)        # /GRSHEL/SHEL/<id> -> [ids]
    surf_grshel: dict = field(default_factory=dict)   # /SURF/GRSHEL/<id> -> [grshel group ids]
    rbody: dict = field(default_factory=dict)         # rbody id -> {"master", "grnd"}
    bcs: list = field(default_factory=list)           # [{"tra", "rot", "grnod"}]
    cloads: list = field(default_factory=list)        # [loaded grnod id]
    inters7: list = field(default_factory=list)       # [{"id", "slav", "mast"}]


def index_deck(lines: list[str]) -> DeckIndex:
    """Single streaming pass over *lines* collecting the load-path topology."""
    idx = DeckIndex()
    mode = cur = None
    captured = False        # single-data-line blocks (rbody/bcs/cload/inter)
    for line in lines:
        if line[:1] == "/":
            s = line.strip()
            captured = False
            cur = None
            if s == "/NODE":
                mode = "node"
            elif s.startswith("/GRNOD/NODE/"):
                mode, cur = "grnod", _tail_id(s, 3)
                if cur is not None:
                    idx.grnod.setdefault(cur, [])
            elif s.startswith("/GRSHEL/SHEL/"):
                mode, cur = "grshel", _tail_id(s, 3)
                if cur is not None:
                    idx.grshel.setdefault(cur, [])
            elif s.startswith("/SURF/GRSHEL/"):
                mode, cur = "surf_grshel", _tail_id(s, -1)
                if cur is not None:
                    idx.surf_grshel.setdefault(cur, [])
            elif s.startswith("/RBODY/"):
                mode, cur = "rbody", _tail_id(s, 2)
                if cur is not None:
                    idx.rbody.setdefault(cur, {})
            elif s.startswith("/BCS/"):
                mode = "bcs"
            elif s.startswith("/CLOAD/"):
                mode = "cload"
            elif s.startswith("/INTER/TYPE7/"):
                mode, cur = "inter7", _tail_id(s, 3)
                idx.inters7.append({"id": cur})
            else:
                mode = None
            continue
        if mode is None or line.lstrip()[:1] == "#" or not line.strip():
            continue
        p = line.split()
        if mode == "node":
            if len(p) >= 4 and _is_int(p[0]):
                idx.coords[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
        elif mode in ("grnod", "grshel", "surf_grshel"):
            if all(_is_int(t) for t in p):        # skip title lines
                dst = {"grnod": idx.grnod, "grshel": idx.grshel,
                       "surf_grshel": idx.surf_grshel}[mode]
                dst[cur].extend(int(t) for t in p)
        elif mode == "rbody" and not captured:
            if _is_int(p[0]) and len(p) >= 6:     # node_ID sens skew Ispher Mass grnd_ID ...
                idx.rbody[cur] = {"master": int(p[0]), "grnd": int(p[5])}
                captured = True
        elif mode == "bcs" and not captured:
            # "Tra rot skew_ID grnod_ID" e.g. "111 111 0 90011"
            if len(p) >= 4 and _is_int(p[0]) and _is_int(p[1]) and _is_int(p[3]):
                idx.bcs.append({"tra": p[0], "rot": p[1], "grnod": int(p[3])})
                captured = True
        elif mode == "cload" and not captured:
            # "funct Dir skew sensor grnod ..." e.g. "1 Y 0 0 90008 1 -0.97815"
            if len(p) >= 5 and _is_int(p[0]) and _is_int(p[4]):
                idx.cloads.append(int(p[4]))
                captured = True
        elif mode == "inter7" and not captured:
            if len(p) >= 2 and _is_int(p[0]) and _is_int(p[1]):
                idx.inters7[-1]["slav"] = int(p[0])
                idx.inters7[-1]["mast"] = int(p[1])
                captured = True
    return idx


def _tail_id(section: str, part: int) -> Optional[int]:
    """Integer id in the ``part``-th ``/``-separated field of a section header
    (``-1`` = last), or ``None`` when it is not an integer."""
    bits = section.split("/")
    try:
        return int(bits[part])
    except (IndexError, ValueError):
        return None


def _is_int(tok: str) -> bool:
    return tok.lstrip("+-").isdigit()


# ------------------------------------------------------------------------- discover
@dataclass
class FastModeTie:
    """The design-node tie sets that give the linear model a load + support path.

    ``load_tie`` are added to the loading rigid body's node group
    (``load_grnod``); ``support_tie`` to the fixed rigid body's node group
    (``fix_grnod``). Both are design-node ids (>= ``design_node_min``).
    """
    load_grnod: int
    fix_grnod: int
    load_tie: np.ndarray
    support_tie: np.ndarray

    def summary(self) -> str:
        return (f"tie load->grnod {self.load_grnod} ({self.load_tie.size} design "
                f"nodes), support->grnod {self.fix_grnod} "
                f"({self.support_tie.size} design nodes)")


def discover_tie(deck: Deck, design_node_min: int,
                 support_thresh: float = DEFAULT_SUPPORT_THRESH,
                 idx: Optional[DeckIndex] = None) -> FastModeTie:
    """Discover the load/support tie sets from *deck* (see the module docstring).

    *idx* may be supplied to reuse an already-parsed :class:`DeckIndex` (the tests
    build a tiny one directly); otherwise the deck's lines are indexed here.
    Raises :class:`FastModeError` when the expected contact topology is absent.
    """
    if idx is None:
        idx = index_deck(deck.lines)

    # loading rigid body: master node carries a /CLOAD
    loaded = set()
    for gr in idx.cloads:
        loaded.update(idx.grnod.get(gr, []))
    if not loaded:
        raise FastModeError(
            "fast mode: no /CLOAD found (or its node group is empty) -- cannot "
            "identify the loading rigid body")
    load_rbs = [d for d in idx.rbody.values() if d.get("master") in loaded]
    if len(load_rbs) != 1:
        raise FastModeError(
            f"fast mode: expected exactly one loading rigid body (its master node "
            f"carries the /CLOAD); found {len(load_rbs)}")
    load_grnod = load_rbs[0]["grnd"]

    # fixed rigid body: master node fully fixed via /BCS 111 111
    fully_fixed = set()
    for b in idx.bcs:
        if b["tra"] == "111" and b["rot"] == "111":
            fully_fixed.update(idx.grnod.get(b["grnod"], []))
    fix_rbs = [d for d in idx.rbody.values() if d.get("master") in fully_fixed]
    if len(fix_rbs) != 1:
        raise FastModeError(
            f"fast mode: expected exactly one fully-fixed rigid body (/BCS 111 111 "
            f"on its master node); found {len(fix_rbs)}")
    fix_grnod = fix_rbs[0]["grnd"]
    fix_owned = set(idx.grnod.get(fix_grnod, []))

    # support contact-master surfaces: /INTER/TYPE7 masters that are /SURF/GRSHEL
    # surfaces whose nodes are (mostly) owned by the fixed rigid body.
    fix_surf_node_ids: set[int] = set()
    for it in idx.inters7:
        mast = it.get("mast")
        if mast not in idx.surf_grshel:
            continue                       # e.g. a SURF/PART/EXT self/loading contact
        ids: list[int] = []
        for g in idx.surf_grshel[mast]:
            ids.extend(idx.grshel.get(g, []))
        present = [n for n in set(ids) if n in idx.coords]
        if not present:
            continue
        owned = sum(1 for n in present if n in fix_owned)
        if owned >= 0.5 * len(present):    # this surface belongs to the fixed body
            fix_surf_node_ids.update(present)
    if not fix_surf_node_ids:
        raise FastModeError(
            "fast mode: no fixed-support contact-master surface found (no "
            "/INTER/TYPE7 master /SURF/GRSHEL owned by the fixed rigid body)")

    # coordinate arrays
    all_ids = np.fromiter(idx.coords.keys(), np.int64, count=len(idx.coords))
    all_xyz = np.array([idx.coords[i] for i in all_ids], dtype=float)
    dmask = all_ids >= design_node_min
    d_ids, d_xyz = all_ids[dmask], all_xyz[dmask]
    if d_ids.size == 0:
        raise FastModeError(
            f"fast mode: no design nodes (id >= {design_node_min}) in the deck")

    from scipy.spatial import cKDTree
    dtree = cKDTree(d_xyz)

    # load side: nearest design node to each loading-rbody node (a pin-in-hole)
    load_grp = set(idx.grnod.get(load_grnod, []))
    lxyz = np.array([idx.coords[n] for n in load_grp if n in idx.coords], dtype=float)
    if lxyz.size == 0:
        raise FastModeError(
            f"fast mode: loading rigid body node group {load_grnod} has no nodes")
    _, li = dtree.query(lxyz, k=1)
    load_tie = np.array([n for n in np.unique(d_ids[li]) if n not in load_grp],
                        dtype=np.int64)

    # support side: design nodes within support_thresh of the fixed support surface
    fxyz = np.array([idx.coords[n] for n in fix_surf_node_ids], dtype=float)
    dist_ds, _ = cKDTree(fxyz).query(d_xyz, k=1)
    sel = dist_ds < support_thresh
    support_tie = np.array([n for n in d_ids[sel] if n not in fix_owned],
                           dtype=np.int64)

    return FastModeTie(load_grnod=int(load_grnod), fix_grnod=int(fix_grnod),
                       load_tie=load_tie, support_tie=support_tie)


# --------------------------------------------------------------------------- build
def build_fast_case(deck: Deck, alive: np.ndarray, starter_path: str | Path,
                    src_engine: str | Path, dst_engine: str | Path,
                    tie: FastModeTie, anim_dt: Optional[float] = None) -> dict:
    """Turn the just-written alive starter into a tied-linear fast-mode deck pair.

    Call **after** :meth:`oropt.deck.Deck.write` has written *starter_path* for this
    iteration's alive mask. Injects the tie sets (intersected with the nodes the
    alive mesh actually references, so a node that dropped out of the design — and
    was therefore already fully pinned by ``Deck.write``'s free-node guard — is not
    also made a rigid slave) into the two rigid-body node groups, and writes a plain
    ``/IMPL/LINEAR`` engine to *dst_engine*.

    Returns ``{"load_tied", "support_tied"}`` counts for logging.
    """
    ref = (np.unique(deck.elem_conn[np.asarray(alive, bool)])
           if np.any(alive) else np.empty(0, np.int64))
    alive_nodes = set(int(x) for x in ref)
    load_add = [int(n) for n in tie.load_tie if int(n) in alive_nodes]
    support_add = [int(n) for n in tie.support_tie if int(n) in alive_nodes]
    _inject_grnod_nodes(starter_path,
                        {tie.load_grnod: load_add, tie.fix_grnod: support_add},
                        deck.newline)
    write_linear_engine(src_engine, dst_engine, anim_dt=anim_dt)
    return {"load_tied": len(load_add), "support_tied": len(support_add)}


def _fmt_id_block(ids: list[int]) -> list[str]:
    """Node ids formatted 10-per-line, right-justified width 10 (Radioss free
    format), matching the converter's ``/GRNOD/NODE`` layout."""
    return ["".join(f"{n:>10}" for n in ids[i:i + 10])
            for i in range(0, len(ids), 10)]


def _inject_grnod_nodes(path: str | Path, inject: dict[int, list[int]],
                        newline: str) -> None:
    """Append node ids to named ``/GRNOD/NODE/<id>`` blocks of the deck at *path*.

    Streams the file, emitting each target group's extra ids just before the block
    ends (the next ``/`` section header). ``inject`` maps group id -> ids to add;
    empty lists are skipped. Rewrites *path* in place.
    """
    inject = {gid: ids for gid, ids in inject.items() if ids}
    if not inject:
        return
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    target: Optional[int] = None
    for line in text.splitlines():
        if line[:1] == "/":
            if target is not None:                 # flush before the next section
                out.extend(_fmt_id_block(inject[target]))
                target = None
            s = line.strip()
            if s.startswith("/GRNOD/NODE/"):
                gid = _tail_id(s, 3)
                if gid in inject:
                    target = gid
            out.append(line)
            continue
        out.append(line)
    if target is not None:                          # target block ran to EOF
        out.extend(_fmt_id_block(inject[target]))
    Path(path).write_text(newline.join(out) + newline, encoding="utf-8", newline="")


def write_linear_engine(src_engine: str | Path, dst_engine: str | Path,
                        anim_dt: Optional[float] = None) -> None:
    """Write a plain ``/IMPL/LINEAR`` engine deck to *dst_engine*.

    Reuses the ``/RUN`` header and termination time from *src_engine* (the case's
    real engine deck) so the run name/Tstop match, then emits one linear implicit
    step with the animation outputs the optimiser reads:
    ``/ANIM/ELEM/VONM`` (peak stress), ``/ANIM/ELEM/ENER`` (the specific-energy
    sensitivity), ``/ANIM/BRICK/TENS/STRESS`` and ``/ANIM/VECT/DISP``. No
    ``/IMPL/PRINT/STIF`` — that was only for the offline matrix study; the stock
    engine handles ``/IMPL/LINEAR`` fine.
    """
    raw = Path(src_engine).read_text(encoding="utf-8", errors="replace")
    nl = "\r\n" if "\r\n" in raw[:8192] else "\n"
    run_line, tstop = _run_header(raw.splitlines())
    dt = tstop if anim_dt is None else _fmt_float(anim_dt)
    body = [
        run_line,
        tstop,
        "#",
        "/TFILE",
        "0.0001",
        "#",
        "/PRINT/-1",
        "#",
        "#-  oropt fast mode: tied linear implicit screen (see oropt/fastmode.py).",
        "#   ONE linear step; von-Mises read from the anim, no stiffness-matrix dump.",
        "/ANIM/DT",
        f"0. {dt}",
        "/ANIM/VECT/DISP",
        "/ANIM/BRICK/TENS/STRESS",
        "/ANIM/ELEM/VONM",
        "/ANIM/ELEM/ENER",
        "#",
        "/IMPL/LINEAR",
        "/IMPL/PRINT/NONL/-1",
        "/IMPL/SOLVER/2",
        "  0 0 0 0",
        "/IMPL/MUMPS/AUTOCORE",
        "/IMPL/DTINI",
        "1",
        "#",
        "/MON/ON",
        "#",
    ]
    Path(dst_engine).write_text(nl.join(body) + nl, encoding="utf-8", newline="")


def _run_header(lines: list[str]) -> tuple[str, str]:
    """The ``/RUN/<name>/<n>`` header line and its termination-time value line."""
    for i, ln in enumerate(lines):
        if ln.strip().startswith("/RUN/"):
            for k in range(i + 1, len(lines)):
                s = lines[k].strip()
                if s and s[:1] != "#":
                    return lines[i].strip(), s
            break
    raise FastModeError(
        "fast mode: engine deck has no /RUN header + termination time to reuse")


def _fmt_float(v: float) -> str:
    """Compact float for the /ANIM/DT line (integers keep a trailing dot, matching
    the converter's ``0. 1.0`` style)."""
    f = float(v)
    return f"{f:g}.0" if f == int(f) else f"{f:g}"

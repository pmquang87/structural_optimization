"""Read and rewrite the converted starter deck (``<stem>_0000.rad``).

Topology optimisation here is pure *element removal*, so the deck editor never
reformats anything: it parses the design part's ``/TETRA4`` block and the first
``/NODE`` block once, then on each iteration re-emits the file **verbatim** minus
the deleted element cards (and minus design nodes that no surviving element
references and no boundary condition pins). Everything else — materials, the
``/SURF/PART/EXT`` contact skin (which OpenRadioss regenerates from the surviving
elements), BCs, rigid bodies — is copied through untouched.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

_GRNOD_RE = re.compile(r"^/GRNOD/NODE/")


def _is_section(line: str) -> bool:
    return line[:1] == "/"


def _is_comment(line: str) -> bool:
    s = line.lstrip()
    return s == "" or s[:1] == "#"


def _first_int(line: str) -> int:
    return int(line.split()[0])


def _parse_node(line: str) -> tuple[int, float, float, float]:
    t = line.split()
    if len(t) >= 4:
        return int(t[0]), float(t[1]), float(t[2]), float(t[3])
    return (int(line[0:10]), float(line[10:30]), float(line[30:50]), float(line[50:70]))


def _parse_elem(line: str) -> tuple[int, int, int, int, int]:
    t = line.split()
    if len(t) >= 5:
        return tuple(int(x) for x in t[:5])  # type: ignore[return-value]
    return tuple(int(line[i:i + 10]) for i in range(0, 50, 10))  # type: ignore[return-value]


def _parse_int_row(line: str, ncols: int) -> list[int]:
    """The first *ncols* integers of an element card (id + connectivity).

    Free (whitespace) format first, falling back to Radioss fixed 10-column
    fields — the same convention as :func:`_parse_elem`, generalised to any
    element arity (5 for a TETRA4, 9 for an 8-node BRICK)."""
    t = line.split()
    if len(t) >= ncols:
        return [int(x) for x in t[:ncols]]
    return [int(line[i:i + 10]) for i in range(0, 10 * ncols, 10)]


class Deck:
    """Parsed starter deck with a mutable alive-element view of the design part."""

    def __init__(self, lines: list[str], newline: str, design_part_id: int,
                 design_node_min: int):
        self.lines = lines
        self.newline = newline
        self.design_part_id = design_part_id
        self.design_node_min = design_node_min

        self._node_lo, self._node_hi = self._find_node_block()
        self._elem_lo, self._elem_hi = self._find_elem_block()

        # design node ids referenced by /GRNOD/NODE/* (BC sets) -> never prune
        self.protected_nodes = self._collect_protected_nodes()

        # compact, card-order arrays for computation + region id maps for fast rewriting
        nids, nxyz, nrid = [], [], []
        for i in range(self._node_lo, self._node_hi):
            ln = self.lines[i]
            if _is_comment(ln):
                nrid.append(-1)
                continue
            nid, x, y, z = _parse_node(ln)
            nids.append(nid); nxyz.append((x, y, z)); nrid.append(nid)
        self.node_ids = np.asarray(nids, dtype=np.int64)
        self.node_xyz = np.asarray(nxyz, dtype=float)
        self._node_region_id = np.asarray(nrid, dtype=np.int64)

        eids, conn, erid = [], [], []
        for i in range(self._elem_lo, self._elem_hi):
            ln = self.lines[i]
            if _is_comment(ln):
                erid.append(-1)
                continue
            eid, n1, n2, n3, n4 = _parse_elem(ln)
            eids.append(eid); conn.append((n1, n2, n3, n4)); erid.append(eid)
        self.elem_ids = np.asarray(eids, dtype=np.int64)
        self.elem_conn = np.asarray(conn, dtype=np.int64)
        self._elem_region_id = np.asarray(erid, dtype=np.int64)

    # ---- construction ------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, design_part_id: int,
             design_node_min: int) -> "Deck":
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        newline = "\r\n" if "\r\n" in raw[:65536] else "\n"
        return cls(raw.splitlines(), newline, design_part_id, design_node_min)

    def _find_node_block(self) -> tuple[int, int]:
        try:
            h = next(i for i, ln in enumerate(self.lines) if ln.strip() == "/NODE")
        except StopIteration:
            raise ValueError("no /NODE block found")
        lo = h + 1
        hi = lo
        while hi < len(self.lines) and not _is_section(self.lines[hi]):
            hi += 1
        return lo, hi

    def _find_elem_block(self) -> tuple[int, int]:
        target = f"/TETRA4/{self.design_part_id}"
        try:
            h = next(i for i, ln in enumerate(self.lines) if ln.strip() == target)
        except StopIteration:
            raise ValueError(f"no {target} block found")
        lo = h + 1
        hi = lo
        # element data ends at the next section (/) or comment (#---/PROPERTIES) line
        while hi < len(self.lines) and not (_is_section(self.lines[hi])
                                            or self.lines[hi].lstrip()[:1] == "#"):
            hi += 1
        return lo, hi

    def _collect_protected_nodes(self) -> frozenset:
        prot: set[int] = set()
        i, n = 0, len(self.lines)
        while i < n:
            if _GRNOD_RE.match(self.lines[i].strip()):
                j = i + 1
                while j < n and not _is_section(self.lines[j]):
                    s = self.lines[j].lstrip()
                    if s and s[:1] != "#":
                        for tok in self.lines[j].split():
                            try:
                                v = int(tok)
                            except ValueError:
                                continue
                            if v >= self.design_node_min:
                                prot.add(v)
                    j += 1
                i = j
            else:
                i += 1
        return frozenset(prot)

    # ---- info --------------------------------------------------------------
    @property
    def n_design_elements(self) -> int:
        return int(self.elem_ids.size)

    def max_node_id(self) -> int:
        """Highest node id across **all** ``/NODE`` blocks of the deck.

        ``node_ids`` deliberately exposes only the first block (the design
        part's, the one the optimiser rewrites), but converter output can
        carry further ``/NODE`` blocks — one per source include, e.g. the
        rigid parts' nodes — whose ids may exceed the first block's.
        Generated-node id allocation must clear every block, or the starter
        rejects the extended deck with *NODE ID=... HAS BEEN DECLARED
        MULTIPLE TIMES* (and, the duplicate's coordinates differing, folds
        the elements on the original node into zero/negative volumes)."""
        top = 0
        i, n = 0, len(self.lines)
        while i < n:
            if self.lines[i].strip() != "/NODE":
                i += 1
                continue
            j = i + 1
            while j < n and not _is_section(self.lines[j]):
                if not _is_comment(self.lines[j]):
                    nid = _parse_node(self.lines[j])[0]
                    if nid > top:
                        top = nid
                j += 1
            i = j
        return top

    def group_ids(self) -> list[int]:
        """Ids of every ``/GRNOD/NODE/<id>`` block in the deck (sorted, unique).

        :meth:`group_nodes` silently returns empty for an id with no matching
        block, so the run-start guard (:func:`oropt.loop.validate_group_ids`)
        checks configured group ids against this list and can name the deck's
        real ids when one is mistyped."""
        ids: set[int] = set()
        for ln in self.lines:
            s = ln.strip()
            if _GRNOD_RE.match(s):
                try:
                    ids.add(int(s.split("/")[3]))
                except (IndexError, ValueError):
                    continue
        return sorted(ids)

    def group_nodes(self, group_id: int) -> np.ndarray:
        """Node ids listed in a specific ``/GRNOD/NODE/<group_id>`` block."""
        target = f"/GRNOD/NODE/{group_id}"
        ids: list[int] = []
        n = len(self.lines)
        try:
            i = next(k for k in range(n) if self.lines[k].strip() == target)
        except StopIteration:
            return np.empty(0, dtype=np.int64)
        j = i + 1
        while j < n and not _is_section(self.lines[j]):
            s = self.lines[j].lstrip()
            if s[:1].isdigit():                      # node-id line (skip title/comments)
                try:
                    vals = [int(t) for t in self.lines[j].split()]
                except ValueError:
                    pass    # a digit-leading TITLE (e.g. "2nd_symmetry_set"), not data
                else:
                    ids.extend(vals)
            j += 1
        return np.asarray(ids, dtype=np.int64)

    def _find_header(self, prefix: str) -> int:
        """Index of the first line whose section header is *prefix* (optionally
        with a trailing ``/unit_ID``), or ``-1``."""
        for k, ln in enumerate(self.lines):
            s = ln.strip()
            if s == prefix or s.startswith(prefix + "/"):
                return k
        return -1

    def _box_block(self, start: int
                   ) -> tuple[list, Optional[int], Optional[float]]:
        """Parse the body of a ``/BOX/...`` card starting one line after *start*.

        Returns ``(points, skew_id, diam)``: the coordinate points (each line with
        three or more numeric tokens ``Xp Yp Zp``), plus the ``skew_ID`` and
        ``Diam`` read from the single ``skew_ID [Diam]`` line (one or two numeric
        tokens). The title and any comment lines are skipped, so layout order does
        not matter."""
        pts: list[tuple[float, float, float]] = []
        skew_id: Optional[int] = None
        diam: Optional[float] = None
        n = len(self.lines)
        j = start + 1
        title_skipped = False
        while j < n and not _is_section(self.lines[j]):
            if not _is_comment(self.lines[j]):
                if not title_skipped:
                    # The card's mandatory title line. Skipped unconditionally:
                    # a purely NUMERIC title (e.g. "1234") would otherwise be
                    # consumed as the skew_ID [Diam] line, dropping the real one
                    # (box() then reports the card missing -- or worse, attaches
                    # an unrelated /SKEW frame to the box).
                    title_skipped = True
                    j += 1
                    continue
                toks = self.lines[j].split()
                try:
                    vals = [float(t) for t in toks]
                except ValueError:
                    vals = []                        # non-numeric junk line
                if len(vals) >= 3:
                    pts.append((vals[0], vals[1], vals[2]))
                elif vals and skew_id is None:       # 1-2 tokens: skew_ID [Diam]
                    skew_id = int(vals[0])
                    if len(vals) >= 2:
                        diam = vals[1]
            j += 1
        return pts, skew_id, diam

    def skew_fix(self, skew_id: int) -> Optional[tuple]:
        """Local frame of a ``/SKEW/FIX/<skew_id>`` card as
        ``(origin, x_axis, xy_axis)`` — each a ``[x, y, z]`` list — or ``None``.

        Reads the block's first three coordinate lines as the frame origin, the
        local ``+x`` direction and a vector in the local ``+xy`` plane
        (Gram-Schmidt-orthonormalised downstream, in :func:`oropt.mesh.local_frame_basis`),
        so a ``/BOX/RECTA`` that references this skew becomes an *oriented* growth
        box (LS-DYNA ``*DEFINE_BOX_LOCAL`` -> ``/BOX/RECTA`` + ``/SKEW/FIX``)."""
        i = self._find_header(f"/SKEW/FIX/{skew_id}")
        if i < 0:
            return None
        pts, _, _ = self._box_block(i)
        if len(pts) < 3:
            return None
        return [list(pts[0]), list(pts[1]), list(pts[2])]

    def box(self, box_id: int) -> Optional[dict]:
        """Resolve a ``/BOX/{RECTA,CYLIN,SPHER}/<box_id>`` card to a growth-region
        spec (a dict of :class:`~oropt.config.GrowthBox` fields), or ``None`` when
        no such card is present.

        Lets a region authored in the pre-processor travel with the model and be
        referenced by id from the config
        (:attr:`~oropt.config.GrowthBox.deck_box_id`) instead of literal
        coordinates, for every shape:

        * ``/BOX/RECTA`` -> ``{"shape": "box", x_min..z_max}`` (two corner points,
          normalised); a non-zero ``skew_ID`` referencing a ``/SKEW/FIX`` card
          attaches the local frame (an oriented box);
        * ``/BOX/SPHER`` -> ``{"shape": "sphere", cx, cy, cz, radius}`` (centre
          point + ``Diam``/2);
        * ``/BOX/CYLIN`` -> ``{"shape": "cylinder", x1..z2, radius}`` (two axis
          end-points + ``Diam``/2).

        The header may carry a trailing ``/unit_ID``."""
        for kind in ("RECTA", "SPHER", "CYLIN"):
            i = self._find_header(f"/BOX/{kind}/{box_id}")
            if i < 0:
                continue
            pts, skew_id, diam = self._box_block(i)
            if kind == "RECTA":
                if len(pts) < 2:
                    return None
                (ax, ay, az), (bx, by, bz) = pts[0], pts[1]
                spec = {"shape": "box",
                        "x_min": min(ax, bx), "x_max": max(ax, bx),
                        "y_min": min(ay, by), "y_max": max(ay, by),
                        "z_min": min(az, bz), "z_max": max(az, bz)}
                if skew_id:
                    frame = self.skew_fix(skew_id)
                    if frame is not None:
                        spec["origin"], spec["x_axis"], spec["xy_axis"] = frame
                return spec
            if kind == "SPHER":
                if not pts or diam is None:
                    return None
                cx, cy, cz = pts[0]
                return {"shape": "sphere", "cx": cx, "cy": cy, "cz": cz,
                        "radius": diam / 2.0}
            if kind == "CYLIN":
                if len(pts) < 2 or diam is None:
                    return None
                (x1, y1, z1), (x2, y2, z2) = pts[0], pts[1]
                return {"shape": "cylinder", "x1": x1, "y1": y1, "z1": z1,
                        "x2": x2, "y2": y2, "z2": z2, "radius": diam / 2.0}
        return None

    def box_recta(self, box_id: int) -> Optional[tuple]:
        """Two opposite corners of a ``/BOX/RECTA/<box_id>`` card as
        ``(x_min, x_max, y_min, y_max, z_min, z_max)`` (normalised), or ``None``.

        A thin rectangular-only view over :meth:`box` (kept for callers that only
        want axis-aligned bounds); use :meth:`box` for the full shape set."""
        spec = self.box(box_id)
        if spec is None or spec.get("shape") != "box":
            return None
        return (spec["x_min"], spec["x_max"], spec["y_min"], spec["y_max"],
                spec["z_min"], spec["z_max"])

    # ---- growth-mesh extension ----------------------------------------------
    def extended_lines(self, node_lines: list[str], elem_lines: list[str]) -> list[str]:
        """A copy of :attr:`lines` with *node_lines* appended at the end of the
        first ``/NODE`` block and *elem_lines* at the end of the design
        ``/TETRA4`` block — the deck-splicing seam of the growth-mesh PREPARE
        step (:mod:`oropt.growthmesh`). Everything else is carried through
        verbatim (the same principle as :meth:`write`); this deck object is not
        modified — parse a new :class:`Deck` from the returned lines to see the
        extended mesh."""
        out = list(self.lines)
        for pos, block in sorted([(self._node_hi, node_lines),
                                  (self._elem_hi, elem_lines)], reverse=True):
            out[pos:pos] = list(block)
        return out

    # ---- rewrite -----------------------------------------------------------
    def write(self, out_path: str | Path, alive_mask: np.ndarray,
              no_pin: Optional[set] = None,
              free_group_id: int = 91000001, free_bcs_id: int = 91000002) -> dict:
        """Write *out_path* keeping only alive design elements.

        Nodes are left in ``/NODE`` untouched (the contact slave node-groups and
        ``/SURF/PART/EXT`` master stay valid). A design *mesh* node that no surviving
        element references would otherwise make the implicit tangent singular, so
        it is fully constrained via an injected ``/GRNOD/NODE`` + ``/BCS`` before
        ``/END`` — the converter's free-node guard, generalised. Nodes in *no_pin*
        (already kinematically constrained, e.g. the symmetry set) are skipped to
        avoid conflicting constraints.

        Only nodes that belong to *some* design element are pin candidates. Design-
        range nodes that are element-free in the full mesh — e.g. the converter's
        synthesised ``--rigid-cog-master`` /RBODY master nodes — are structural and
        already kinematically handled by their rigid body; pinning one with a full-
        fix /BCS double-constrains it (locking a loaded master's free DOFs -> zero
        external work) and OpenRadioss flags it as an incompatible kinematic
        condition (WARNING 312 -> AUTOSPC -> dead solve). They are never pinned.

        ``alive_mask`` is a boolean array aligned with :attr:`elem_ids`.
        """
        alive_mask = np.asarray(alive_mask, dtype=bool)
        if alive_mask.shape != self.elem_ids.shape:
            raise ValueError("alive_mask must align with elem_ids")

        ref_nodes = np.unique(self.elem_conn[alive_mask]) if alive_mask.any() \
            else np.empty(0, dtype=np.int64)
        # Only nodes carried by some design element can be orphaned by deletion;
        # element-free design-range nodes (synthesised rigid-body masters) are
        # structural and must not be pinned (see the docstring).
        mesh_nodes = np.unique(self.elem_conn) if self.elem_conn.size \
            else np.empty(0, dtype=np.int64)
        design_nodes = np.intersect1d(
            self.node_ids[self.node_ids >= self.design_node_min], mesh_nodes)
        free_nodes = np.setdiff1d(design_nodes, ref_nodes, assume_unique=False)
        if no_pin:
            free_nodes = free_nodes[~np.isin(free_nodes, np.fromiter(
                no_pin, dtype=np.int64, count=len(no_pin)))]

        out: list[str] = list(self.lines[:self._elem_lo])
        ci = 0
        for j in range(self._elem_lo, self._elem_hi):
            if self._elem_region_id[j - self._elem_lo] == -1:
                out.append(self.lines[j])             # comment/blank -> verbatim
            else:
                if alive_mask[ci]:
                    out.append(self.lines[j])
                ci += 1
        out.extend(self.lines[self._elem_hi:])

        if free_nodes.size:
            self._inject_free_node_constraint(out, free_nodes, free_group_id, free_bcs_id)

        text = self.newline.join(out) + self.newline
        Path(out_path).write_text(text, encoding="utf-8", newline="")
        return {
            "elements_alive": int(alive_mask.sum()),
            "elements_total": int(alive_mask.size),
            "free_nodes_pinned": int(free_nodes.size),
        }

    @staticmethod
    def _inject_free_node_constraint(out: list[str], free_nodes: np.ndarray,
                                     grp_id: int, bcs_id: int) -> None:
        """Insert a fully-fixed /GRNOD/NODE + /BCS for *free_nodes* before /END."""
        block = [f"/GRNOD/NODE/{grp_id}", "oropt_free_nodes"]
        ids = free_nodes.tolist()
        for k in range(0, len(ids), 10):
            block.append("".join(f"{v:>10}" for v in ids[k:k + 10]))
        block += [
            f"/BCS/{bcs_id}", "oropt_free_fix",
            "#  Tra rot   skew_ID  grnod_ID",
            f"   111 111         0{grp_id:>10}",
        ]
        end = next((i for i in range(len(out) - 1, -1, -1)
                    if out[i].strip() == "/END"), len(out))
        out[end:end] = block


def prepare_engine(src_engine: str | Path, dst_engine: str | Path,
                   anim_dt: Optional[float] = None) -> None:
    """Copy the engine deck, optionally reducing animation output frequency.

    The baseline deck writes 11 animation states (~811 MB / run); the optimiser
    only needs the final converged state. Setting ``anim_dt`` >= the termination
    time makes OpenRadioss emit just the start + end states, slashing I/O.
    Only the ``/ANIM/DT`` value line is touched; the implicit controls are left
    exactly as the converter wrote them.
    """
    raw = Path(src_engine).read_text(encoding="utf-8", errors="replace")
    nl = "\r\n" if "\r\n" in raw[:8192] else "\n"
    lines = raw.splitlines()
    if anim_dt is not None:
        for i, ln in enumerate(lines):
            if ln.strip() == "/ANIM/DT":
                # next non-comment line is "Tstart  dt"
                k = i + 1
                while k < len(lines) and lines[k].lstrip()[:1] == "#":
                    k += 1
                if k < len(lines):
                    toks = lines[k].split()
                    tstart = toks[0] if toks else "0."
                    lines[k] = f"{tstart} {anim_dt}"
                break
    Path(dst_engine).write_text(nl.join(lines) + nl, encoding="utf-8", newline="")


# --------------------------------------------------------------------------- #
# neighbour-part geometry (keep-out decks)
#
# A keep-out deck (:mod:`oropt.keepout`) describes nearby parts that are never
# solved -- only their occupied volume matters, as forbidden growth space. Unlike
# the design :class:`Deck` (one design part, the first /NODE block), a keep-out
# deck may carry several parts across several /NODE blocks (converter output emits
# one /NODE block per include), so this reads ALL of them.
# --------------------------------------------------------------------------- #

#: solid element blocks read for keep-out geometry, mapped to their node count.
_SOLID_HEADER_RE = re.compile(r"^/(TETRA4|BRICK)/(\d+)\s*$")

#: 6-tet decomposition of an 8-node /BRICK (indices into its n1..n8), sharing the
#: n1-n7 main diagonal — exact for a convex hex, which any solver brick is.
_HEX_TO_TETS = np.array([
    [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
    [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]], dtype=np.int64)

#: the four triangular faces of a TETRA4 (indices into its n1..n4)
_TET4_FACES = np.array(
    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
#: the six quad faces of an 8-node /BRICK (Radioss n1-n4 bottom, n5-n8 top).
#: Boundary detection runs on these PRE-SPLIT quads: after the 6-tet split, two
#: adjacent bricks can triangulate their shared quad along different diagonals,
#: so the split triangles would not cancel and an interior brick-brick interface
#: would read as "surface". A shared quad always cancels as a quad.
_HEX_FACES = np.array([
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=np.int64)


def _boundary_node_ids(conn: np.ndarray, face_idx: np.ndarray) -> np.ndarray:
    """Node ids on the boundary of a conforming solid mesh *conn*: the nodes of
    every face (per *face_idx*) used by exactly ONE element. Same face-counting
    idea as :func:`oropt.growthmesh.exterior_faces`, generalised to quads."""
    if not len(conn):
        return np.empty(0, dtype=np.int64)
    faces = conn[:, face_idx].reshape(-1, face_idx.shape[1])
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True,
                                   return_counts=True)
    return np.unique(faces[counts[inverse] == 1])


def read_solid_geometry(path: str | Path, part_ids=None):
    """Parse a Radioss deck for the solid-element geometry of the named parts.

    Reads EVERY ``/NODE`` block (a converted deck emits one per include) and every
    ``/TETRA4/<pid>`` and ``/BRICK/<pid>`` block whose part id is in *part_ids*
    (all solid parts when *part_ids* is ``None``/empty). Returns
    ``(tet_xyz, node_xyz, surf_xyz, part_ids)`` where:

    * ``tet_xyz`` is ``(V, 4, 3)`` — the parts' occupied volume as a set of
      tetrahedra (each brick split into 6 tets), for a point-in-volume test;
    * ``node_xyz`` is ``(P, 3)`` — the coordinates of every node the selected
      elements reference (for the positive-clearance distance test);
    * ``surf_xyz`` is ``(S, 3)`` — the coordinates of the nodes on the parts'
      boundary SURFACE only (faces used by exactly one element; brick faces are
      matched as pre-split quads so adjacent bricks' mismatched split diagonals
      cannot fake a surface). The negative-clearance (allowed-penetration)
      depth test needs these: the full node cloud includes interior nodes,
      whose small distances would masquerade as "near the surface" deep inside
      the volume. At a mixed tet↔brick part interface the (unmatchable)
      tri-vs-quad faces both read as surface — such interface nodes are rare
      and only shrink the measured depth there;
    * ``part_ids`` is the sorted list of solid part ids actually found.

    Raises ``ValueError`` when the deck holds no matching solid elements or a
    solid element references a node id absent from every ``/NODE`` block."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    n = len(lines)
    want = {int(p) for p in part_ids} if part_ids else None

    # pass 1: every node across every /NODE block
    node_map: dict[int, tuple[float, float, float]] = {}
    i = 0
    while i < n:
        if lines[i].strip() == "/NODE":
            i += 1
            while i < n and not _is_section(lines[i]):
                if not _is_comment(lines[i]):
                    nid, x, y, z = _parse_node(lines[i])
                    node_map[nid] = (x, y, z)
                i += 1
            continue
        i += 1

    # pass 2: the selected solid element blocks. Raw (pre-split) connectivity is
    # kept per element type so the surface can be detected on real element
    # faces; bricks are split to tets only for the volume test.
    tet4_conn: list = []
    hex_conn: list = []
    found: set[int] = set()
    i = 0
    while i < n:
        m = _SOLID_HEADER_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        etype, pid = m.group(1), int(m.group(2))
        ncols = 5 if etype == "TETRA4" else 9
        selected = want is None or pid in want
        i += 1
        block: list = []
        while i < n and not (_is_section(lines[i])
                             or lines[i].lstrip()[:1] == "#"):
            if not _is_comment(lines[i]):
                block.append(_parse_int_row(lines[i], ncols))
            i += 1
        if not (selected and block):
            continue
        found.add(pid)
        if etype == "TETRA4":
            tet4_conn.extend(row[1:5] for row in block)
        else:
            hex_conn.extend(row[1:9] for row in block)

    if not (tet4_conn or hex_conn):
        where = f" for part id(s) {sorted(want)}" if want else ""
        raise ValueError(f"no solid (/TETRA4 or /BRICK) elements found in "
                         f"{path}{where}")

    tet4 = np.asarray(tet4_conn, dtype=np.int64).reshape(-1, 4)
    hexes = np.asarray(hex_conn, dtype=np.int64).reshape(-1, 8)
    tet_conn = (np.concatenate([tet4, hexes[:, _HEX_TO_TETS].reshape(-1, 4)])
                if len(hexes) else tet4)               # bricks -> 6 tets each

    uniq = np.unique(tet_conn)
    try:
        coords = np.array([node_map[int(v)] for v in uniq], dtype=float)
    except KeyError as exc:
        raise ValueError(f"keep-out solid element references node id {exc} absent "
                         f"from every /NODE block in {path}") from exc
    rows = np.searchsorted(uniq, tet_conn.ravel()).reshape(tet_conn.shape)
    tet_xyz = coords[rows]                              # (V, 4, 3)

    surf_ids = np.union1d(_boundary_node_ids(tet4, _TET4_FACES),
                          _boundary_node_ids(hexes, _HEX_FACES))
    surf_xyz = coords[np.searchsorted(uniq, surf_ids)]  # (S, 3)
    return tet_xyz, coords, surf_xyz, sorted(found)

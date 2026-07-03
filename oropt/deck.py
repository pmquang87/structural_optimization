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
                ids.extend(int(t) for t in self.lines[j].split())
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
        while j < n and not _is_section(self.lines[j]):
            if not _is_comment(self.lines[j]):
                toks = self.lines[j].split()
                try:
                    vals = [float(t) for t in toks]
                except ValueError:
                    vals = []                        # non-numeric -> the title
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

    # ---- rewrite -----------------------------------------------------------
    def write(self, out_path: str | Path, alive_mask: np.ndarray,
              no_pin: Optional[set] = None,
              free_group_id: int = 91000001, free_bcs_id: int = 91000002) -> dict:
        """Write *out_path* keeping only alive design elements.

        Nodes are left in ``/NODE`` untouched (the contact slave node-groups and
        ``/SURF/PART/EXT`` master stay valid). Design nodes that no surviving
        element references would otherwise make the implicit tangent singular, so
        they are fully constrained via an injected ``/GRNOD/NODE`` + ``/BCS``
        before ``/END`` — the converter's free-node guard, generalised. Nodes in
        *no_pin* (already kinematically constrained, e.g. the symmetry set) are
        skipped to avoid conflicting constraints.

        ``alive_mask`` is a boolean array aligned with :attr:`elem_ids`.
        """
        alive_mask = np.asarray(alive_mask, dtype=bool)
        if alive_mask.shape != self.elem_ids.shape:
            raise ValueError("alive_mask must align with elem_ids")

        ref_nodes = np.unique(self.elem_conn[alive_mask]) if alive_mask.any() \
            else np.empty(0, dtype=np.int64)
        design_nodes = self.node_ids[self.node_ids >= self.design_node_min]
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

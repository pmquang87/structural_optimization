"""Extract per-element and nodal results from OpenRadioss output.

``anim_to_vtk`` converts the last animation state to a legacy VTK that carries,
per solid cell, ``ELEMENT_ID`` / ``PART_ID`` / ``3DELEM_Specific_Energy``
(BESO sensitivity) / ``3DELEM_Von_Mises`` (stress), and per node ``NODE_ID`` /
``Displacement``. That single conversion yields every quantity the optimiser
needs, so the hot path never touches ``th_to_csv``.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config
from .runner import build_env, find_last_anim

# VTK array names emitted by anim_to_vtk for 4-node solids (confirmed against this model).
F_ELEMENT_ID = "ELEMENT_ID"
F_PART_ID = "PART_ID"
F_EROSION = "EROSION_STATUS"
F_ENERGY = "3DELEM_Specific_Energy"
F_VONMISES = "3DELEM_Von_Mises"
P_NODE_ID = "NODE_ID"
P_DISP = "Displacement"

# Sentinel for extract()'s disp_node_id default, so an explicit None (a case that
# tracks no displacement node) is distinct from "argument omitted".
_UNSET = object()


@dataclass
class Results:
    """Design-part element fields (aligned arrays) plus scalar constraints.

    ``disps`` maps every requested constrained node id -> its |displacement|
    [mm] (``nan`` for a node absent from the animation). ``disp`` / ``disp_node_id``
    are the first requested node's convenience scalars (``nan`` / ``None`` when no
    node is tracked), kept so single-node callers stay unchanged.
    """
    element_ids: np.ndarray      # int64, design-part solid elements still present
    energy: np.ndarray           # float, specific (internal) energy per element  -> BESO sensitivity
    vonmises: np.ndarray         # float, von-Mises per element [MPa]
    sigma_max: float             # peak von-Mises over the design part [MPa]
    disp: float                  # |displacement| at the first constrained node [mm]
    disp_node_id: Optional[int]
    disps: dict = field(default_factory=dict)   # {node_id: |displacement| [mm]} for every constrained node

    def as_dict(self) -> dict:
        return {"sigma_max": self.sigma_max, "disp": self.disp,
                "n_elem": int(self.element_ids.size)}

    @property
    def is_null_solve(self) -> bool:
        """True when the design part carried no load at all: not one element
        developed positive von-Mises stress *or* positive strain energy.

        A loaded solid always develops stress somewhere, so an identically-zero
        response means the load never reached the mesh -- the force landed on a
        constrained / rigid DOF, a contact interface never engaged, or the deck
        was mis-exported. The loop uses this to fail loudly instead of
        "optimising" a dead model: with every field zero, ``sigma_max``/``disp``
        read 0 and pass every feasibility limit trivially, and the BESO-family
        energy sensitivity is uniformly zero, so the optimiser would strip the
        part down to its protected skeleton (exactly how ``opti_run5_Ti`` lost
        continuity). Note an empty design part (no elements) is also reported
        null -- a degenerate state equally unfit to optimise."""
        has_stress = bool(self.vonmises.size) and bool(np.any(self.vonmises > 0.0))
        has_energy = bool(self.energy.size) and bool(np.any(self.energy > 0.0))
        return not (has_stress or has_energy)


def run_anim_to_vtk(cfg: Config, anim_file: Path, out_vtk: Path) -> Path:
    """Convert one animation file to VTK (written to *out_vtk*)."""
    exe = cfg.or_paths.abs("anim_to_vtk")
    if not exe.exists():
        raise FileNotFoundError(f"anim_to_vtk not found: {exe}")
    env = build_env(cfg)
    with open(out_vtk, "w", encoding="utf-8", errors="replace") as fh:
        cp = subprocess.run([str(exe), str(anim_file)], stdout=fh,
                            stderr=subprocess.PIPE, env=env, check=False)
    if cp.returncode != 0 or out_vtk.stat().st_size < 1024:
        raise RuntimeError(f"anim_to_vtk failed for {anim_file.name}: "
                           f"{cp.stderr.decode(errors='replace')[:300]}")
    return out_vtk


def _node_id_list(disp_node_id, disp_node_ids) -> list[int]:
    """Normalise the two accepted node arguments to a list of ids.

    *disp_node_ids* (the multi-node list) wins when given; otherwise the legacy
    scalar *disp_node_id* becomes a one- or zero-entry list. ``None`` -> no node
    tracked.
    """
    if disp_node_ids is not None:
        return [int(n) for n in disp_node_ids]
    return [] if disp_node_id is None else [int(disp_node_id)]


def parse_vtk(vtk_path: Path, design_part_id: int,
              disp_node_id: Optional[int] = None,
              disp_node_ids: Optional[list] = None,
              exclude_element_ids: Optional[np.ndarray] = None) -> Results:
    """Read the VTK and pull out design-part solid fields + the constrained node(s).

    Read with pyvista (VTK's own reader) — robust to the field names anim_to_vtk
    emits (``/``, ``&``, spaces) that trip simpler parsers. ``cell_data`` is global
    per cell, so filtering on ``PART_ID`` isolates the design solids directly.

    Pass *disp_node_ids* (a list) to read several nodes' displacements at once — a
    load case may constrain several nodes, each with its own limit — or the legacy
    scalar *disp_node_id* for a single node. The animation already contains every
    nodal displacement, so reading many is free. Every requested node lands in
    ``Results.disps`` (``nan`` if absent from the animation).

    ``exclude_element_ids`` is the stress-exclusion set (design elements touching a
    known hot-spot the user flagged): their von-Mises is dropped from ``sigma_max``
    so it never drives the feasibility verdict or the Monitor/report peak. The
    per-element ``energy``/``vonmises`` arrays stay full (the sensitivity still sees
    every element); only the reported peak stress excludes them.
    """
    import pyvista as pv
    grid = pv.read(str(vtk_path))

    def cell_arr(name):
        if name not in grid.cell_data:
            raise KeyError(f"cell field {name!r} missing from {vtk_path.name}")
        return np.asarray(grid.cell_data[name])

    pid = cell_arr(F_PART_ID).astype(np.int64)
    keep = pid == design_part_id
    if F_EROSION in grid.cell_data:                    # EROSION_STATUS: 1=active, 0=eroded
        keep &= cell_arr(F_EROSION).astype(int) == 1
    eid = cell_arr(F_ELEMENT_ID).astype(np.int64)[keep]
    energy = cell_arr(F_ENERGY).astype(float)[keep]
    vm = cell_arr(F_VONMISES).astype(float)[keep]
    vm_rated = vm                                       # von-Mises that counts toward sigma_max
    if exclude_element_ids is not None and len(exclude_element_ids):
        excl = np.isin(eid, np.asarray(exclude_element_ids, dtype=np.int64))
        vm_rated = vm[~excl]
    sigma_max = float(vm_rated.max()) if vm_rated.size else float("nan")

    node_list = _node_id_list(disp_node_id, disp_node_ids)
    disps: dict = {}
    if node_list and P_NODE_ID in grid.point_data:
        all_ids = np.asarray(grid.point_data[P_NODE_ID]).astype(np.int64)
        dvec = np.asarray(grid.point_data[P_DISP])
        for nid in node_list:
            loc = np.where(all_ids == nid)[0]
            disps[nid] = (float(np.linalg.norm(dvec[loc[0]])) if loc.size
                          else float("nan"))
    else:
        disps = {nid: float("nan") for nid in node_list}

    first = node_list[0] if node_list else None
    disp = disps.get(first, float("nan")) if first is not None else float("nan")
    return Results(element_ids=eid, energy=energy, vonmises=vm,
                   sigma_max=sigma_max, disp=disp, disp_node_id=first, disps=disps)


def extract(cfg: Config, run_dir: str | Path, keep_vtk: bool = False,
            stem: Optional[str] = None, disp_node_id=_UNSET,
            disp_node_ids: Optional[list] = None,
            exclude_element_ids: Optional[np.ndarray] = None) -> Results:
    """Convert the latest animation in *run_dir* and parse it into Results.

    *stem* selects which case's animation/VTK to read; *disp_node_ids* (a list) or
    the legacy scalar *disp_node_id* select which node(s)' displacement to report.
    The loop passes each case's own values. When *stem* / node arguments are
    omitted they default to the primary (first) load case's stem and displacement
    constraint node(s); an explicit ``disp_node_id=None`` (or empty
    ``disp_node_ids``) is respected (the case tracks no displacement node).
    *exclude_element_ids* is forwarded to :func:`parse_vtk` so the stress-exclusion
    region is dropped from the reported ``sigma_max``.
    """
    run_dir = Path(run_dir)
    if stem is None:
        stem = cfg.primary_case().stem
    if disp_node_ids is None and disp_node_id is _UNSET:
        disp_node_ids = [dc.node_id for dc in cfg.primary_case().disp_constraints]
    if disp_node_id is _UNSET:
        disp_node_id = None
    anim = find_last_anim(run_dir, stem)
    if anim is None:
        raise FileNotFoundError(f"no animation file <{stem}A0NN> in {run_dir}")
    out_vtk = run_dir / f"{stem}_last.vtk"
    run_anim_to_vtk(cfg, anim, out_vtk)
    res = parse_vtk(out_vtk, cfg.model.design_part_id, disp_node_id=disp_node_id,
                    disp_node_ids=disp_node_ids,
                    exclude_element_ids=exclude_element_ids)
    if not keep_vtk:
        out_vtk.unlink(missing_ok=True)
    return res

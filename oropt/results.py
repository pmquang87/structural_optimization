"""Extract per-element and nodal results from OpenRadioss output.

``anim_to_vtk`` converts the last animation state to a legacy VTK that carries,
per solid cell, ``ELEMENT_ID`` / ``PART_ID`` / ``3DELEM_Specific_Energy``
(BESO sensitivity) / ``3DELEM_Von_Mises`` (stress), and per node ``NODE_ID`` /
``Displacement``. That single conversion yields every quantity the optimiser
needs, so the hot path never touches ``th_to_csv``.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
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


@dataclass
class Results:
    """Design-part element fields (aligned arrays) plus scalar constraints."""
    element_ids: np.ndarray      # int64, design-part solid elements still present
    energy: np.ndarray           # float, specific (internal) energy per element  -> BESO sensitivity
    vonmises: np.ndarray         # float, von-Mises per element [MPa]
    sigma_max: float             # peak von-Mises over the design part [MPa]
    disp: float                  # |displacement| at the constrained node [mm]
    disp_node_id: Optional[int]

    def as_dict(self) -> dict:
        return {"sigma_max": self.sigma_max, "disp": self.disp,
                "n_elem": int(self.element_ids.size)}


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


def parse_vtk(vtk_path: Path, design_part_id: int,
              disp_node_id: Optional[int]) -> Results:
    """Read the VTK and pull out design-part solid fields + the constrained node.

    Read with pyvista (VTK's own reader) — robust to the field names anim_to_vtk
    emits (``/``, ``&``, spaces) that trip simpler parsers. ``cell_data`` is global
    per cell, so filtering on ``PART_ID`` isolates the design solids directly.
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
    sigma_max = float(vm.max()) if vm.size else float("nan")

    disp = float("nan")
    if disp_node_id is not None and P_NODE_ID in grid.point_data:
        node_ids = np.asarray(grid.point_data[P_NODE_ID]).astype(np.int64)
        loc = np.where(node_ids == int(disp_node_id))[0]
        if loc.size:
            d = np.asarray(grid.point_data[P_DISP])[loc[0]]
            disp = float(np.linalg.norm(d))

    return Results(element_ids=eid, energy=energy, vonmises=vm,
                   sigma_max=sigma_max, disp=disp, disp_node_id=disp_node_id)


def extract(cfg: Config, run_dir: str | Path, keep_vtk: bool = False) -> Results:
    """Convert the latest animation in *run_dir* and parse it into Results."""
    run_dir = Path(run_dir)
    anim = find_last_anim(run_dir, cfg.model.stem)
    if anim is None:
        raise FileNotFoundError(f"no animation file <{cfg.model.stem}A0NN> in {run_dir}")
    out_vtk = run_dir / f"{cfg.model.stem}_last.vtk"
    run_anim_to_vtk(cfg, anim, out_vtk)
    res = parse_vtk(out_vtk, cfg.model.design_part_id, cfg.model.disp_node_id)
    if not keep_vtk:
        out_vtk.unlink(missing_ok=True)
    return res

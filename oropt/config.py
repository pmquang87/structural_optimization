"""Configuration for an oropt BESO run (YAML-backed dataclass).

A single :class:`Config` captures everything needed to reproduce a run: where the
OpenRadioss install and the converted deck live, how to launch the solver, which
part is the design domain, the stress/displacement limits, and the BESO knobs.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ORPaths:
    """Locations of the OpenRadioss install and its executables/post tools."""
    root: str = r"C:\OpenRadioss"
    starter: str = "exec/starter_win64.exe"
    engine: str = "exec/engine_win64_impi.exe"     # MPI engine, launched via mpiexec -np 1 (proven path)
    anim_to_vtk: str = "exec/anim_to_vtk_win64.exe"
    th_to_csv: str = "exec/th_to_csv_win64.exe"
    cfg_path: str = "hm_cfg_files"                 # -> RAD_CFG_PATH
    h3d_path: str = r"extlib\h3d\lib\win64"         # -> RAD_H3D_PATH
    # Intel oneAPI MPI: mpiexec + impi.dll live in <root>/bin; libfabric under opt/mpi/libfabric/bin.
    intel_mpi_root: str = r"C:\Program Files (x86)\Intel\oneAPI\mpi\latest"

    def abs(self, attr: str) -> Path:
        return (Path(self.root) / getattr(self, attr)).resolve()

    def mpiexec(self) -> Path:
        return (Path(self.intel_mpi_root) / "bin" / "mpiexec.exe").resolve()

    def mpi_path_dirs(self) -> list[str]:
        r = Path(self.intel_mpi_root)
        return [str(r / "bin"), str(r / "opt" / "mpi" / "libfabric" / "bin")]


@dataclass
class RunOpts:
    """How to invoke the solver and how patient to be."""
    np: int = 1                # MUST be 1: SPMD implicit + solid contact segfaults (documented)
    nt: int = 6                # OpenMP threads (i9-13900H livelock mitigation -> 6)
    use_mpi: bool = True       # launch engine via mpiexec -np N (the bare engine fails to load its MPI DLLs)
    starter_timeout_s: float = 600.0
    engine_timeout_s: float = 3600.0
    kmp_stacksize: str = "400m"
    anim_dt: float = 1.0       # animation output interval; >= termination time -> only the final state


@dataclass
class Model:
    """The converted deck and what it contains."""
    case_dir: str = "."                 # folder holding <stem>_0000.rad / _0001.rad
    stem: str = "implicit_elevator-linkage"
    design_part_id: int = 60000000      # /TETRA4/<id> block to optimise
    design_node_min: int = 60000000     # design nodes have ids >= this (rigid parts are 10xxxxxx)
    disp_node_id: Optional[int] = None  # node whose displacement is constrained (default: load rigid-body master)
    bc_group_id: int = 60000000         # /GRNOD/NODE/<id> holding the BC/symmetry nodes to protect
    # User-defined keep-out / non-design regions: any design element touching one
    # of these nodes is frozen (never deleted). Give /GRNOD/NODE/<id> group ids
    # (e.g. 99999999) and/or explicit node ids.
    freeze_group_ids: list = field(default_factory=list)
    freeze_node_ids: list = field(default_factory=list)

    def starter(self) -> Path:
        return Path(self.case_dir).resolve() / f"{self.stem}_0000.rad"

    def engine(self) -> Path:
        return Path(self.case_dir).resolve() / f"{self.stem}_0001.rad"


@dataclass
class Constraints:
    """High-fidelity feasibility limits, checked against OpenRadioss output."""
    sigma_allow: float = 250.0   # max von-Mises [MPa]
    d_allow: float = 1.0         # max |displacement| at disp_node_id [mm]


@dataclass
class Beso:
    """BESO algorithm knobs."""
    evolution_rate: float = 0.02     # ER: target volume fraction removed per iteration
    max_add_ratio: float = 0.01      # AR_max: cap on re-added (resurrected) volume per iteration
    filter_radius: float = 1.5       # spatial sensitivity-filter radius [mm] (~2-3x element size)
    history_weight: float = 0.5      # blend of current & previous-iteration sensitivity (0.5 = Huang-Xie)
    target_volume_fraction: float = 0.5  # stop reducing once this fraction of the design volume remains
    sensitivity: str = "energy"      # "energy" | "vonmises" | "blend"
    blend_weight: float = 0.5        # weight on von-Mises when sensitivity == "blend"
    max_iter: int = 150
    convergence_tol: float = 1e-3    # rel. change in objective over the averaging window
    convergence_window: int = 5
    protect_layers: int = 2          # element layers around protected nodes to freeze (never delete)
    contact_protect_dist: float = 0.0  # also protect design elements within this distance of a rigid (cylinder) node


@dataclass
class Config:
    or_paths: ORPaths = field(default_factory=ORPaths)
    run: RunOpts = field(default_factory=RunOpts)
    model: Model = field(default_factory=Model)
    constraints: Constraints = field(default_factory=Constraints)
    beso: Beso = field(default_factory=Beso)
    work_dir: str = "runs/run01"     # per-iteration scratch + checkpoints + status files

    # ---- (de)serialisation -------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        def build(klass, sub):
            return klass(**{k: v for k, v in (sub or {}).items()
                            if k in {f.name for f in dataclasses.fields(klass)}})
        return cls(
            or_paths=build(ORPaths, data.get("or_paths")),
            run=build(RunOpts, data.get("run")),
            model=build(Model, data.get("model")),
            constraints=build(Constraints, data.get("constraints")),
            beso=build(Beso, data.get("beso")),
            work_dir=data.get("work_dir", "runs/run01"),
        )

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(asdict(self), sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )

    def work(self) -> Path:
        p = Path(self.work_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

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
class DockerOpts:
    """Optionally run OpenRadioss via the Dockerised MUMPS-implicit build instead
    of the native Windows binaries — no Intel oneAPI/MKL/MPI install needed, and
    it works on AMD or Intel. The run folder is bind-mounted to ``/data`` and the
    container writes its outputs (.out, A0NN, T01, .rst) back there, so the rest
    of the pipeline is unchanged. See the image's ``COLLEAGUE_INSTRUCTIONS.md``.

    When enabled, ``or_paths`` and the Intel-MPI ``run`` settings are ignored; the
    container supports real MPI, so ``np`` may be > 1 (keep ``np * nt`` <= cores).
    """
    enabled: bool = False
    image: str = "openradioss-mumps:20260520"
    docker_exe: str = "docker"       # docker CLI: a name on PATH or a full path
    shm_size: str = "2g"             # --shm-size (MUMPS needs shared memory)
    np: int = 4                      # MPI domains (Docker build supports np > 1)
    nt: int = 1                      # OpenMP threads per domain
    extra_args: list = field(default_factory=list)  # extra `docker run` args, e.g. ["--cpus", "8"]


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
    protect_bc_nodes: bool = True    # freeze elements touching the BC node-group (model.bc_group_id). False -> they may be deleted; the BC nodes stay fixed via their /BCS and still anchor connectivity
    archive_iterations: bool = True    # keep each iteration's deck/anim/listing in work_dir/iter_NNNN/ (on by default; see README disk cost)
    archive_restart: bool = False      # when archiving, also copy the ~345 MB restart (<stem>*.rst) into iter_NNNN/ -> full per-iteration solver state. OFF by default: ~345 MB/iter would be ~50 GB over a 150-iter run; opt in when you need replayable solver state


@dataclass
class LevelSet:
    """Discrete nodal level-set optimiser knobs (a config-selectable alternative
    to BESO that yields smoother boundaries than ragged element removal).

    The structure is represented by a nodal level-set field phi; an element is
    alive iff its nodes' mean phi >= 0. Each iteration scatters the filtered
    per-element energy onto nodes (a "velocity" Vn), evolves
    ``phi <- phi + dt*(Vn - lambda)`` with ``lambda`` found by bisection so the
    thresholded volume hits the per-iteration target, then runs a few Laplacian
    smoothing passes (reaction-diffusion-style regularisation) for smooth
    boundaries and clamps phi to +/-``band_width`` to stay bounded.

    The first block mirrors the BESO knobs of the same name (so a level-set run is
    fully specified by its own config block, the way ``run``/``docker`` each carry
    their own ``np``/``nt``); the second block is level-set specific.
    """
    # --- shared semantics with BESO ---
    evolution_rate: float = 0.02     # ER: target volume fraction removed per iteration
    filter_radius: float = 1.5       # spatial sensitivity-filter radius [mm]
    history_weight: float = 0.5      # blend of current & previous-iteration sensitivity
    target_volume_fraction: float = 0.5  # stop reducing once this volume fraction remains
    sensitivity: str = "energy"      # "energy" | "vonmises" | "blend"
    blend_weight: float = 0.5        # weight on von-Mises when sensitivity == "blend"
    max_iter: int = 150
    convergence_tol: float = 1e-3    # rel. change in objective over the averaging window
    convergence_window: int = 5
    protect_layers: int = 2          # element layers around protected nodes to freeze
    contact_protect_dist: float = 0.0  # also protect design elements within this distance of a rigid node
    protect_bc_nodes: bool = True    # freeze elements touching the BC node-group
    archive_iterations: bool = True    # keep each iteration's deck/anim/listing in work_dir/iter_NNNN/ (on by default)
    archive_restart: bool = False      # when archiving, also copy the restart (.rst). OFF by default (~345 MB/iter); opt in for replayable solver state
    # --- level-set specific ---
    dt: float = 1.0                  # pseudo-time step for the phi evolution
    smoothing_passes: int = 3        # Laplacian/Jacobi smoothing passes per iteration (regularisation)
    band_width: float = 3.0          # clamp |phi| to this after each step to keep the field bounded


@dataclass
class TobsOpts:
    """TOBS (Topology Optimisation of Binary Structures) optimiser knobs — a
    config-selectable alternative to BESO (Sivapuram & Picelli, *Finite Elements
    in Analysis and Design* 139:49-61, 2018).

    The design variables are the same binary alive/void element flags as BESO, but
    each iteration the *flips* ``dx_e in {-1,0,+1}`` are chosen by solving a small
    0/1 integer linear program (``scipy.optimize.milp`` / HiGHS) rather than by a
    sensitivity threshold:

    * **objective** — maximise ``sum_e s_e * dx_e`` (keep high-, drop
      low-sensitivity elements), with ``s_e`` the filtered/​history-averaged
      strain-energy density shared with BESO;
    * **move limit** — at most a fraction ``flip_limit`` of all elements may flip
      per iteration (``sum_e |dx_e| <= flip_limit * N``);
    * **volume constraint** — the element-volume-weighted volume is stepped toward
      the per-iteration target (``evolution_rate``/``target_volume_fraction``, same
      gate as BESO) as a linearised constraint relaxed by ``constraint_relaxation``
      so the binary subproblem is always feasible.

    Protected elements are forced to stay alive and disconnected islands are
    dropped, exactly like BESO. The first block mirrors the BESO/level-set shared
    knobs (so a TOBS run is fully specified by its own config block); the second
    block is TOBS specific.
    """
    # --- shared semantics with BESO ---
    evolution_rate: float = 0.02     # ER: target volume fraction removed per iteration
    filter_radius: float = 1.5       # spatial sensitivity-filter radius [mm]
    history_weight: float = 0.5      # blend of current & previous-iteration sensitivity
    target_volume_fraction: float = 0.5  # stop reducing once this volume fraction remains
    sensitivity: str = "energy"      # "energy" | "vonmises" | "blend"
    blend_weight: float = 0.5        # weight on von-Mises when sensitivity == "blend"
    max_iter: int = 150
    convergence_tol: float = 1e-3    # rel. change in objective over the averaging window
    convergence_window: int = 5
    protect_layers: int = 2          # element layers around protected nodes to freeze
    contact_protect_dist: float = 0.0  # also protect design elements within this distance of a rigid node
    protect_bc_nodes: bool = True    # freeze elements touching the BC node-group
    archive_iterations: bool = True    # keep each iteration's deck/anim/listing in work_dir/iter_NNNN/ (on by default)
    archive_restart: bool = False      # when archiving, also copy the restart (.rst). OFF by default (~345 MB/iter); opt in for replayable solver state
    # --- TOBS specific ---
    flip_limit: float = 0.05         # beta: max fraction of elements flipped per ILP step (Sum|dx| <= beta*N)
    constraint_relaxation: float = 0.01  # epsilon: relaxation band (x V0) on the linearised volume constraint


@dataclass
class D3plotOpts:
    """Optional post-run conversion of the final OpenRadioss animation into an
    LS-Dyna ``d3plot`` (viewable in LS-PrePost etc.).

    Delegated to the external Vortex-Radioss ``Anim_to_D3plot`` tool, run as an
    *isolated subprocess* so its dependency set (lasso-python, tqdm) never has to
    be installed alongside oropt. Strictly best-effort: a missing tool, missing
    interpreter or a failed conversion is logged and skipped — it never aborts or
    fails the optimisation run.
    """
    enabled: bool = True
    # Folder containing the ``vortex_radioss`` package (the openradioss_tools repo
    # root); placed on the converter subprocess's ``sys.path``. Blank -> the
    # ``OROPT_VORTEX_ROOT`` environment variable (so the default config stays
    # portable across machines instead of hard-coding one user's checkout).
    tool_root: str = ""
    # Interpreter that has lasso-python/tqdm installed. Blank -> ``<tool_root>/
    # .venv`` if present, else the interpreter running oropt.
    python_exe: str = ""
    show_rigidwall: bool = True      # keep rigid-wall/rigid parts in the d3plot
    timeout_s: float = 1800.0        # cap on the conversion subprocess


@dataclass
class SmoothOpts:
    """Surface smoothing of the optimised geometry (on by default).

    When enabled, after a run finishes the surface of the final design (the
    latest ``topology_latest.vtu``) is extracted, smoothed and written as
    ``topology_smoothed.<ext>`` in the run folder — a clean deliverable for
    CAD / 3D-print / review — and every per-iteration snapshot is smoothed too
    (``topology_smoothed_iterNNNN.<ext>``). Best-effort: a failure is logged,
    never fatal.
    """
    enabled: bool = True
    iterations: int = 20             # smoothing passes
    method: str = "taubin"           # "taubin" (volume-preserving) | "laplacian" (shrinks)
    pass_band: float = 0.1           # Taubin pass-band (smaller -> smoother)
    relaxation: float = 0.1          # Laplacian relaxation factor (method == "laplacian")
    output_format: str = "stl"       # "stl" | "vtp" | "both"


@dataclass
class ReportOpts:
    """Automatic post-run summary report of the optimisation.

    When enabled (default — it's cheap and read-only), after a run finishes a
    human-readable summary is written into the run folder: ``report.html`` (a
    self-contained file with the convergence charts and a render of the final
    design embedded) and ``report.md``. It only *reads* the artefacts the loop
    already wrote (``status.json``, ``history.csv``, ``topology_latest.vtu``), so
    it never touches the run. Best-effort: a missing/failing matplotlib (charts)
    or pyvista (topology render) degrades gracefully to file links and is logged,
    never fatal. The render runs in an *isolated subprocess* (like ``d3plot``) so
    even a hard GL/driver crash on a headless box is contained and never aborts
    the run. See :mod:`oropt.report`.
    """
    enabled: bool = True
    charts: bool = True              # matplotlib convergence charts (vf, sigma, disp)
    render_topology: bool = True     # off-screen pyvista render of the final design
    render_timeout_s: float = 120.0  # cap on the isolated render subprocess


@dataclass
class CustomView:
    """A user-defined, reusable camera angle for the evolution animation.

    A custom view is a built-in preset (``base``) plus ``azimuth`` / ``elevation``
    offsets in degrees, saved under a ``name`` so it can be re-selected by that name
    in :attr:`AnimateOpts.view` (and the GUI's angle dropdown). It lets a user dial
    in a favourite three-quarter angle once and reuse it, instead of re-entering the
    offsets each run. The global :attr:`AnimateOpts.azimuth` / ``elevation`` still
    apply on top as a final nudge.
    """
    name: str = ""
    base: str = "iso"        # built-in preset: iso|front|back|left|right|top|bottom
    azimuth: float = 0.0     # azimuth offset [deg] from the base preset
    elevation: float = 0.0   # elevation offset [deg] from the base preset


@dataclass
class AnimateOpts:
    """Automatic post-run animation of the topology evolution (on by default).

    When enabled, after a run finishes the per-iteration smoothed surfaces
    (``topology_smoothed_iterNNNN.stl`` — produced by :class:`SmoothOpts`, falling
    back to the raw ``topology_iterNNNN.vtu`` snapshots when smoothing is off) are
    rendered from a *single fixed camera* and assembled into
    ``topology_evolution.gif`` in the run folder — a quick visual of material being
    removed across the optimisation. The camera angle is chosen with ``view``: a
    built-in preset (``iso`` / ``front`` / ``back`` / ``left`` / ``right`` / ``top``
    / ``bottom``) *or* the ``name`` of one of the user-defined :class:`CustomView`
    entries in ``custom_views``. It is fine-tuned with ``azimuth`` / ``elevation``
    offsets (degrees), so any viewpoint is reachable; whichever angle is picked it
    stays *fixed* across every frame so the design loses material in place. Like the
    report's topology render, the frames
    are drawn by an *isolated off-screen pyvista subprocess* (so a hard GL/driver
    crash on a headless box is contained and never aborts the run) and the GIF is
    encoded with Pillow in-process. Best-effort: a missing/failing dependency or a
    run with fewer than two snapshots is logged and skipped, never fatal. See
    :mod:`oropt.animate`.
    """
    enabled: bool = True
    fps: float = 4.0                 # frames per second of the output GIF
    view: str = "iso"                # built-in preset name OR a custom_views name
    azimuth: float = 0.0             # extra camera azimuth rotation [deg] after the preset
    elevation: float = 0.0           # extra camera elevation rotation [deg] after the preset
    window_w: int = 900              # render width  [px]
    window_h: int = 600              # render height [px]
    color: str = "lightsteelblue"    # solid surface colour of the design
    opacity: float = 1.0             # surface opacity 0..1 (1 = solid, <1 = see-through)
    background: str = "white"        # frame background colour
    show_edges: bool = False         # draw mesh edges on the surface
    show_labels: bool = True         # stamp "iter N" on each frame
    hold_last: int = 6               # linger on the final design (×frame duration)
    render_timeout_s: float = 300.0  # cap on the isolated render subprocess (all frames)
    # User-defined named angles selectable via ``view``. Stored as CustomView, but
    # coerced from plain dicts too so YAML round-trips and the GUI editor (which
    # produces dict rows) both work without a special case in Config.from_dict.
    custom_views: list = field(default_factory=list)

    def __post_init__(self):
        fields = {f.name for f in dataclasses.fields(CustomView)}
        self.custom_views = [
            v if isinstance(v, CustomView)
            else CustomView(**{k: val for k, val in dict(v).items() if k in fields})
            for v in (self.custom_views or [])]


@dataclass
class ManufacturingOpts:
    """Additive-manufacturing (AM) printability constraints applied to the alive
    mask each iteration, *after* the optimiser's own update (so they work for
    BESO and the level-set alike). The target part is powder-bed-fusion printed
    (e.g. AlSi10Mg), so these keep the evolving topology manufacturable.

    All fields default to OFF, so existing runs are byte-identical. Applied in
    order by :func:`oropt.manufacturing.apply_manufacturing`:

    1. **Minimum member size** — a morphological *open* (erode then dilate over
       shared-node element adjacency) deletes thin features / single-element
       slivers thinner than the structuring element. ``min_member_layers`` is the
       number of erode/dilate hops (0 = off; 1-2 is typical).
    2. **Symmetry planes** — force the design symmetric across each plane. Rule:
       *either alive ⇒ both alive* (an element is kept if it or its mirror is
       alive), so symmetry is enforced without over-removing and volume control
       catches up over iterations. Each entry is a mapping
       ``{"axis": "x"|"y"|"z", "offset": <plane coordinate>}``.
    3. **Overhang / self-support** — along ``build_direction`` forbid any alive
       element that has no solid support within a downward cone of half-angle
       ``max_overhang_angle`` degrees (measured from the build direction); the
       lowest layer rests on the build plate. ``build_direction = None`` or
       ``max_overhang_angle <= 0`` turns it off.
    """
    min_member_layers: int = 0
    # list of {"axis": "x"|"y"|"z", "offset": float}
    symmetry_planes: list = field(default_factory=list)
    build_direction: Optional[list] = None   # [x, y, z]; None -> overhang off
    max_overhang_angle: float = 0.0          # cone half-angle [deg] from build dir; 0 -> off


@dataclass
class LoadCase:
    """One load case: a *separate* deck pair that shares the design mesh but
    applies a different load (the elevator linkage pulled in another direction).

    The model is deliberately the simplest one that reuses the whole solve path
    unchanged: a load case is identified by its own deck ``stem`` (its
    ``<stem>_0000.rad`` starter + ``<stem>_0001.rad`` engine in ``model.case_dir``)
    whose *only* meaningful difference from the others is the applied-load cards
    (``/CLOAD`` / imposed motion / etc.). All cases MUST share the same
    design-part element ids and node ids — element removal is identical across
    cases, so each iteration writes the same alive set into every case's deck and
    only the load cards differ.

    Blank/None fields fall back to the single-case defaults: ``stem`` ->
    ``model.stem``, ``disp_node_id`` -> ``model.disp_node_id``, ``sigma_allow`` /
    ``d_allow`` -> the global ``constraints``. The combined sensitivity is the
    per-case-normalised weighted sum ``sum_i weight_i * energy_i`` and a design is
    feasible only when *every* case is feasible against its own limits.
    """
    name: str = "default"
    stem: str = ""                       # source deck stem; blank -> model.stem
    weight: float = 1.0                  # w_i in the weighted-sum sensitivity
    disp_node_id: Optional[int] = None   # blank -> model.disp_node_id
    sigma_allow: Optional[float] = None  # blank -> constraints.sigma_allow
    d_allow: Optional[float] = None      # blank -> constraints.d_allow


@dataclass
class ResolvedCase:
    """A :class:`LoadCase` with every fallback filled in and deck paths resolved.

    Runtime-only (never serialised): produced by :meth:`Config.load_case_list`.
    """
    name: str
    stem: str
    weight: float
    disp_node_id: Optional[int]
    sigma_allow: float
    d_allow: float
    starter: Path
    engine: Path


@dataclass
class Config:
    or_paths: ORPaths = field(default_factory=ORPaths)
    run: RunOpts = field(default_factory=RunOpts)
    docker: DockerOpts = field(default_factory=DockerOpts)
    model: Model = field(default_factory=Model)
    constraints: Constraints = field(default_factory=Constraints)
    beso: Beso = field(default_factory=Beso)
    levelset: LevelSet = field(default_factory=LevelSet)
    tobs: TobsOpts = field(default_factory=TobsOpts)
    manufacturing: ManufacturingOpts = field(default_factory=ManufacturingOpts)
    d3plot: D3plotOpts = field(default_factory=D3plotOpts)
    smooth: SmoothOpts = field(default_factory=SmoothOpts)
    report: ReportOpts = field(default_factory=ReportOpts)
    animate: AnimateOpts = field(default_factory=AnimateOpts)
    # Multiple load cases (optional). Leave empty for the classic single-case run
    # (one implicit case == the ``model`` deck, weight 1) — behaviour is then
    # byte-identical to before. List ``LoadCase`` entries to minimise a
    # weighted-sum compliance over several loads. See :class:`LoadCase`.
    load_cases: list = field(default_factory=list)
    # Which topology optimiser to drive the loop: "beso" (default, bi-directional
    # element removal), "levelset" (nodal level-set, smoother boundaries) or "tobs"
    # (binary ILP flips, Sivapuram & Picelli 2018). The active block's shared knobs
    # (target_volume_fraction, max_iter, convergence, protect_*, archive_*) are read
    # via ``active_opts()``.
    optimizer: str = "beso"
    # Run/output folder: per-iteration scratch + checkpoints + status files. Leave
    # blank to default to the input deck folder (``model.case_dir``) itself; set a
    # path (e.g. ``runs/run01``) to put outputs elsewhere.
    work_dir: str = ""

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
            docker=build(DockerOpts, data.get("docker")),
            model=build(Model, data.get("model")),
            constraints=build(Constraints, data.get("constraints")),
            beso=build(Beso, data.get("beso")),
            levelset=build(LevelSet, data.get("levelset")),
            tobs=build(TobsOpts, data.get("tobs")),
            manufacturing=build(ManufacturingOpts, data.get("manufacturing")),
            d3plot=build(D3plotOpts, data.get("d3plot")),
            smooth=build(SmoothOpts, data.get("smooth")),
            report=build(ReportOpts, data.get("report")),
            animate=build(AnimateOpts, data.get("animate")),
            load_cases=[build(LoadCase, lc) for lc in (data.get("load_cases") or [])],
            optimizer=(data.get("optimizer") or "beso"),
            work_dir=data.get("work_dir") or "",
        )

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(asdict(self), sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )

    @staticmethod
    def read_yaml_dict(path: str | Path) -> dict:
        """The raw mapping parsed from *path* (``{}`` for an empty file).

        Exposed so callers can validate the *as-written* config (e.g. flag
        unrecognised keys via :func:`unknown_keys`) before they are silently
        dropped by :meth:`from_dict`.
        """
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    # ---- optimiser selection ----------------------------------------------
    def optimizer_name(self) -> str:
        """Normalised optimiser selector: ``"beso"``, ``"levelset"`` or ``"tobs"``."""
        return (self.optimizer or "beso").strip().lower()

    def active_opts(self):
        """The config block for the selected optimiser. The loop reads the
        run-level knobs shared by the optimisers (target_volume_fraction,
        max_iter, convergence_*, protect_*, archive_*) from here so it stays
        optimiser-agnostic."""
        name = self.optimizer_name()
        if name == "levelset":
            return self.levelset
        if name == "tobs":
            return self.tobs
        return self.beso

    def load_case_list(self) -> list[ResolvedCase]:
        """Resolve :attr:`load_cases` into concrete cases with fallbacks applied.

        With no configured cases this returns the single implicit case (the
        ``model`` deck, weight 1) so the optimiser's multi-case path collapses to
        exactly the classic single-solve behaviour.
        """
        m, c = self.model, self.constraints
        case_dir = Path(m.case_dir).resolve()
        specs = self.load_cases or [
            LoadCase(name="default", stem=m.stem, weight=1.0,
                     disp_node_id=m.disp_node_id,
                     sigma_allow=c.sigma_allow, d_allow=c.d_allow)]
        out: list[ResolvedCase] = []
        for lc in specs:
            stem = lc.stem or m.stem
            out.append(ResolvedCase(
                name=lc.name or stem,
                stem=stem,
                weight=float(lc.weight),
                disp_node_id=(lc.disp_node_id if lc.disp_node_id is not None
                              else m.disp_node_id),
                sigma_allow=(lc.sigma_allow if lc.sigma_allow is not None
                             else c.sigma_allow),
                d_allow=(lc.d_allow if lc.d_allow is not None else c.d_allow),
                starter=case_dir / f"{stem}_0000.rad",
                engine=case_dir / f"{stem}_0001.rad"))
        return out

    def run_folder(self) -> str:
        """The configured run/output folder *as written* (may be relative).

        Falls back to the input deck folder (``model.case_dir``) itself when
        ``work_dir`` is blank, so by default a run writes its status/history/
        topology right next to the decks it was built from. The mutated deck still
        goes to ``<run_folder>/solve/`` — a sub-folder — so the source decks are
        never clobbered.
        """
        wd = (self.work_dir or "").strip()
        if wd:
            return wd
        return str(Path(self.model.case_dir))

    def work(self) -> Path:
        p = Path(self.run_folder()).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


def _field_names(klass) -> set[str]:
    return {f.name for f in dataclasses.fields(klass)}


def _section_types() -> dict[str, type]:
    """``{section name: dataclass}`` for every nested-dataclass field of Config.

    Derived from Config's own fields (each section is a ``field(default_factory=
    <DataclassType>)``) so it can never drift out of sync with :meth:`Config.from_dict`
    as sections are added. Scalar fields (``optimizer``, ``work_dir``) and the
    ``load_cases`` list are not dataclass sections and are handled separately.
    """
    out: dict[str, type] = {}
    for f in dataclasses.fields(Config):
        factory = f.default_factory  # type: ignore[misc]
        if factory is not dataclasses.MISSING:
            inst = factory()
            if dataclasses.is_dataclass(inst):
                out[f.name] = type(inst)
    return out


def unknown_keys(data: dict) -> list[str]:
    """Config keys in the raw mapping *data* that :meth:`Config.from_dict` ignores.

    :meth:`Config.from_dict` silently drops any key it does not recognise, so a
    typo (``evolution_ratte``) or a knob placed under the wrong section reverts to
    the default — an expensive surprise on a multi-hour run. This walks *data*
    against the known schema (top-level sections/scalars, each section's fields,
    and the ``load_cases`` / ``animate.custom_views`` lists) and returns a
    dotted-path name for every unrecognised key, so validation can warn about them.
    """
    if not isinstance(data, dict):
        return []
    sections = _section_types()
    out: list[str] = []

    top_known = set(sections) | {"load_cases", "optimizer", "work_dir"}
    out.extend(k for k in data if k not in top_known)

    for name, klass in sections.items():
        sub = data.get(name)
        if isinstance(sub, dict):
            known = _field_names(klass)
            out.extend(f"{name}.{k}" for k in sub if k not in known)

    def _list_of(parent_key: str, items, klass) -> None:
        if isinstance(items, list):
            known = _field_names(klass)
            for i, row in enumerate(items):
                if isinstance(row, dict):
                    out.extend(f"{parent_key}[{i}].{k}"
                               for k in row if k not in known)

    _list_of("load_cases", data.get("load_cases"), LoadCase)
    anim = data.get("animate")
    if isinstance(anim, dict):
        _list_of("animate.custom_views", anim.get("custom_views"), CustomView)

    return out

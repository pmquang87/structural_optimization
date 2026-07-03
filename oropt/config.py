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
    nt: int = 12               # OpenMP threads to solve with (np stays 1); livelock is mitigated by KMP_BLOCKTIME=0 / OMP_WAIT_POLICY=PASSIVE in runner.build_env, not by capping threads
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
class GrowthBox:
    """A user-defined axis-aligned box (two opposite corners, like LS-DYNA's
    ``*DEFINE_BOX`` / Radioss ``/BOX/RECTA``) marking part of the design mesh as
    **candidate growth material**: design elements whose centroid lies inside any
    growth box start the run *void* (removed from the deck) and may be **added**
    by the optimiser's bi-directional update wherever the load path wants them —
    letting the design grow material where the original part had none.

    The box volume must be **pre-meshed** into the design part (same
    ``/TETRA4/<design_part_id>`` block, node-conformal interface with the original
    part, node ids >= ``design_node_min``); a box over unmeshed space selects no
    elements and is an error at run start. Multiple boxes act as a union. Bounds
    are inclusive; a box overlapping the original part volume voids those
    elements at start too (deliberate carve-and-regrow).
    """
    name: str = ""
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0


@dataclass
class Model:
    """The shared geometry of the design: which part is the design domain and
    which nodes anchor it. The deck identity (``stem``) and the constrained
    displacement node live per load case (see :class:`LoadCase`), not here, so a
    single-load run is just one load case."""
    case_dir: str = "."                 # folder holding <stem>_0000.rad / _0001.rad for each load case
    design_part_id: int = 60000000      # /TETRA4/<id> block to optimise
    design_node_min: int = 60000000     # design nodes have ids >= this (rigid parts are 10xxxxxx)
    bc_group_id: int = 60000000         # /GRNOD/NODE/<id> holding the BC/symmetry nodes to protect
    # User-defined keep-out / non-design regions: any design element touching one
    # of these nodes is frozen (never deleted). Give /GRNOD/NODE/<id> group ids
    # (e.g. 99999999) and/or explicit node ids.
    freeze_group_ids: list = field(default_factory=list)
    freeze_node_ids: list = field(default_factory=list)
    # User-defined stress-exclusion regions: any design element touching one of
    # these nodes has its von-Mises *ignored* — left out of the reported peak
    # stress (sigma_max), the feasibility check, and the stress shown in the
    # Monitor/report. Use it for a known hot-spot a later design phase will fix
    # (e.g. around a small cylinder) so it can't drive the optimisation or flag the
    # design infeasible. Give /GRNOD/NODE/<id> group ids (e.g. 999999998) and/or
    # explicit node ids. These elements are NOT frozen — they still take part in
    # the optimisation; add them to freeze_group_ids/freeze_node_ids too if you
    # also want to protect them from removal.
    stress_exclude_group_ids: list = field(default_factory=list)
    stress_exclude_node_ids: list = field(default_factory=list)
    # User-defined growth boxes (add-material regions): design elements whose
    # centroid lies inside any box start VOID and may be grown into by the
    # optimiser. The box volumes must be pre-meshed into the design part; see
    # :class:`GrowthBox`. Stored as GrowthBox, but coerced from plain dicts too
    # so YAML round-trips and the GUI editor (dict rows) both work.
    growth_boxes: list = field(default_factory=list)

    def __post_init__(self):
        fields = {f.name for f in dataclasses.fields(GrowthBox)}
        self.growth_boxes = [
            b if isinstance(b, GrowthBox)
            else GrowthBox(**{k: v for k, v in dict(b).items() if k in fields})
            for b in (self.growth_boxes or [])]


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
    # --- feasibility back-off controller (defaults = the classic binary gate) ---
    backoff_gain: float = 0.0        # proportional back-off: when infeasible, grow by ER*min(gain*(v-1), cap) with v the worst value/limit ratio, instead of a fixed +ER step. 0 = classic binary gate
    backoff_cap: float = 4.0         # cap on the proportional growth step, in multiples of ER (only used when backoff_gain > 0)
    damping_threshold: float = 1.0   # while feasible with v above this, slow removal by (1-v)/(1-threshold) so the design glides into the limit instead of ping-ponging. 1.0 = off (full rate until infeasible)
    addback_stress_bias: float = 0.0  # when a stress limit is violated, scale the update's sensitivity by (1 + bias * filtered vonmises/sigma_allow) so recovered material lands near the overstressed region. 0 = off


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
    # --- feasibility back-off controller (defaults = the classic binary gate) ---
    backoff_gain: float = 0.0        # proportional back-off: when infeasible, grow by ER*min(gain*(v-1), cap) with v the worst value/limit ratio, instead of a fixed +ER step. 0 = classic binary gate
    backoff_cap: float = 4.0         # cap on the proportional growth step, in multiples of ER (only used when backoff_gain > 0)
    damping_threshold: float = 1.0   # while feasible with v above this, slow removal by (1-v)/(1-threshold) so the design glides into the limit instead of ping-ponging. 1.0 = off (full rate until infeasible)
    addback_stress_bias: float = 0.0  # when a stress limit is violated, scale the update's sensitivity by (1 + bias * filtered vonmises/sigma_allow) so recovered material lands near the overstressed region. 0 = off
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
    # --- feasibility back-off controller (defaults = the classic binary gate) ---
    backoff_gain: float = 0.0        # proportional back-off: when infeasible, grow by ER*min(gain*(v-1), cap) with v the worst value/limit ratio, instead of a fixed +ER step. 0 = classic binary gate
    backoff_cap: float = 4.0         # cap on the proportional growth step, in multiples of ER (only used when backoff_gain > 0)
    damping_threshold: float = 1.0   # while feasible with v above this, slow removal by (1-v)/(1-threshold) so the design glides into the limit instead of ping-ponging. 1.0 = off (full rate until infeasible)
    addback_stress_bias: float = 0.0  # when a stress limit is violated, scale the update's sensitivity by (1 + bias * filtered vonmises/sigma_allow) so recovered material lands near the overstressed region. 0 = off
    # --- TOBS specific ---
    flip_limit: float = 0.05         # beta: max fraction of elements flipped per ILP step (Sum|dx| <= beta*N)
    constraint_relaxation: float = 0.01  # epsilon: relaxation band (x V0) on the linearised volume constraint


@dataclass
class HcaOpts:
    """HCA (Hybrid Cellular Automata) optimiser knobs — a config-selectable
    alternative to BESO (Tovar et al., *J. Mech. Des.* 128(6), 2006; the method
    behind LS-TaSC, built for nonlinear/contact problems with no design
    gradients — exactly this regime).

    Every element keeps a continuous *virtual density* ``x_e in [0.01, 1]``
    that persists between iterations. Each iteration a proportional controller
    drives it toward a uniform energy-density setpoint ``S*``
    (``x_e += kp * (S_e - S*)/S*``, move-limited), with ``S*`` found by
    bisection so the thresholded design (alive iff ``x_e >= 0.5``) hits the
    per-iteration volume target (``evolution_rate``/``target_volume_fraction``,
    same gate as BESO). ``S_e`` is the filtered/history-averaged strain-energy
    density shared with BESO — the filter doubles as the cellular automaton's
    neighbourhood averaging.

    Protected elements are pinned at full density and forced alive, and
    disconnected islands are dropped, exactly like BESO. The first block
    mirrors the BESO/level-set/TOBS shared knobs (so an HCA run is fully
    specified by its own config block); the second block is HCA specific.
    """
    # --- shared semantics with BESO ---
    evolution_rate: float = 0.02     # ER: target volume fraction removed per iteration
    filter_radius: float = 1.5       # spatial sensitivity-filter radius [mm] (= the CA neighbourhood)
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
    # --- feasibility back-off controller (defaults = the classic binary gate) ---
    backoff_gain: float = 0.0        # proportional back-off: when infeasible, grow by ER*min(gain*(v-1), cap) with v the worst value/limit ratio, instead of a fixed +ER step. 0 = classic binary gate
    backoff_cap: float = 4.0         # cap on the proportional growth step, in multiples of ER (only used when backoff_gain > 0)
    damping_threshold: float = 1.0   # while feasible with v above this, slow removal by (1-v)/(1-threshold) so the design glides into the limit instead of ping-ponging. 1.0 = off (full rate until infeasible)
    addback_stress_bias: float = 0.0  # when a stress limit is violated, scale the update's sensitivity by (1 + bias * filtered vonmises/sigma_allow) so recovered material lands near the overstressed region. 0 = off
    # --- HCA specific ---
    kp: float = 1.0                  # proportional gain of the density controller
    move_limit: float = 1.0          # cap on |dx_e| per iteration (1.0 = uncapped). Keep min(kp, move_limit) > 0.5 or no element can be removed in a single step
    field_history_weight: float = 1.0  # extra HCA-internal blend of the energy field with previous iterations (LS-TaSC's multi-iteration weighted sum); 1.0 = off (the shared history_weight already blends iterations)


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
    the run.

    ``interactive_topology`` adds a **zoom/rotate** VTK.js viewer of the final
    design to ``report.html`` (the same scene the GUI's Monitor tab shows), via
    pyvista's ``Plotter.export_html``. That needs the optional **trame** export
    backend (``pip install "oropt[report3d]"`` — i.e. ``trame`` / ``trame-vtk``);
    when it isn't installed the report silently falls back to the static
    ``render_topology`` image, so this flag is safe to leave on. See
    :mod:`oropt.report`.
    """
    enabled: bool = True
    charts: bool = True              # matplotlib convergence charts (vf, sigma, disp)
    render_topology: bool = True     # off-screen pyvista PNG of the final design (fallback)
    interactive_topology: bool = True  # zoom/rotate VTK.js viewer in report.html (needs trame)
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
    window_w: int = 1440             # render width  [px]
    window_h: int = 960              # render height [px]
    color: str = "gray"              # solid surface colour of the design
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
    """Manufacturing constraints applied to the alive mask each iteration,
    *after* the optimiser's own update (so they work for BESO, the level-set,
    TOBS and HCA alike). The target part is powder-bed-fusion printed
    (e.g. AlSi10Mg) but may also be cast / extruded, so these keep the evolving
    topology manufacturable. They are not purely additive — casting and extrusion
    can add *or* remove material — so the order of application matters.

    All fields default to OFF, so existing runs are byte-identical. Applied in
    order by :func:`oropt.manufacturing.apply_manufacturing`:

    1. **Minimum member size** (TOSCA/OptiStruct MINGAP-style) — a morphological
       *open* (erode then dilate over shared-node element adjacency) deletes thin
       features / single-element slivers thinner than the structuring element.
       ``min_member_layers`` is the number of erode/dilate hops (0 = off; 1-2 is
       typical). Anti-extensive: only ever removes material.
    2. **Maximum member size** (OptiStruct MAXDIM) — forbid bulky solid lumps:
       every alive element must lie within ``max_member_layers`` adjacency hops
       of a void. Elements deeper than that are carved (least-useful first when a
       sensitivity field is supplied, else geometrically), punching distributed
       voids into thick regions while leaving walls of the allowed thickness.
       0 = off. Protected elements are never carved.
    3. **Symmetry planes** — force the design symmetric across each plane. Rule:
       *either alive ⇒ both alive* (an element is kept if it or its mirror is
       alive), so symmetry is enforced without over-removing and volume control
       catches up over iterations. Each entry is a mapping
       ``{"axis": "x"|"y"|"z", "offset": <plane coordinate>}``.
    4. **Casting / draw direction** (OptiStruct DTPL, TOSCA demold, LS-TaSC
       casting) — along ``draw_direction`` every column of elements must be free
       of undercuts so a die can slide out. *Single-sided* (``draw_two_sided =
       False``): walking up the column, once material ends it may not restart
       (the solid must be a bottom prefix); solids above the first void are
       removed. *Two-sided* (``draw_two_sided = True``): the solid must be one
       contiguous run around a parting surface, so all but the largest run in a
       column is removed. ``draw_direction = None`` turns it off.
    5. **Extrusion** (OptiStruct extrusion constraint) — constant cross-section
       along ``extrusion_axis``: elements are binned into prisms by their
       projected 2-D footprint and each prism is made uniform by a *majority
       vote* (a prism is solid iff at least half of its elements are alive; ties
       kept alive). Majority — rather than symmetry's either-alive ⇒ alive — is
       used because a full-length prism resurrected from a single stray element
       would spike volume and fight the optimiser; the vote tracks the design's
       own intent, and volume control still reconciles over iterations.
       ``extrusion_axis = None`` turns it off.
    6. **Overhang / self-support** — along ``build_direction`` forbid any alive
       element that has no solid support within a downward cone of half-angle
       ``max_overhang_angle`` degrees (measured from the build direction); the
       lowest layer rests on the build plate. Applied *last* so support is judged
       on the near-final mask. ``build_direction = None`` or
       ``max_overhang_angle <= 0`` turns it off.

    Protected elements (BC/load/keep-out) always survive — they are OR'd back in
    at the end — so an enabled constraint may leave a residual feature around a
    protected region (the user's keep-out choice). When several directional
    constraints are combined they can conflict (e.g. a draw direction that fights
    a symmetry plane); the later one wins that iteration and volume control
    reconciles across iterations. Disconnected islands a constraint may create
    are re-dropped by the caller (:mod:`oropt.loop` via ``mesh.keep_connected``).
    """
    min_member_layers: int = 0
    max_member_layers: int = 0               # MAXDIM hops-to-void limit; 0 -> off
    # list of {"axis": "x"|"y"|"z", "offset": float}
    symmetry_planes: list = field(default_factory=list)
    draw_direction: Optional[list] = None    # [x, y, z] casting draw dir; None -> off
    draw_two_sided: bool = False             # False = single-sided prefix; True = one run around a parting surface
    extrusion_axis: Optional[list] = None    # [x, y, z] constant cross-section axis; None -> off
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

    A load case is the **single source of truth** for the deck ``stem`` and the
    feasibility limits ``sigma_allow`` / ``d_allow`` (and the constrained
    ``disp_node_id``): every run defines at least one. The classic single-load run
    is simply one load case. The combined sensitivity is the per-case-normalised
    weighted sum ``sum_i weight_i * energy_i`` and a design is feasible only when
    *every* case is feasible against its own limits.

    ``stem`` and the two limits are required (a blank value is a validation
    error). ``disp_node_id`` is optional: ``None`` means no displacement node is
    tracked for this case.
    """
    name: str = "default"
    stem: str = ""                       # source deck stem (<stem>_0000.rad / _0001.rad); required
    weight: float = 1.0                  # w_i in the weighted-sum sensitivity
    disp_node_id: Optional[int] = None   # constrained node; None -> no disp node tracked
    sigma_allow: Optional[float] = None  # max von-Mises [MPa]; required
    d_allow: Optional[float] = None      # max |displacement| at disp_node_id [mm]; required


@dataclass
class ResolvedCase:
    """A :class:`LoadCase` with every fallback filled in and deck paths resolved.

    Runtime-only (never serialised): produced by :meth:`Config.load_case_list`.
    """
    name: str
    stem: str
    weight: float
    disp_node_id: Optional[int]
    sigma_allow: Optional[float]    # None only for an unvalidated config (validation requires it)
    d_allow: Optional[float]        # None only for an unvalidated config (validation requires it)
    starter: Path
    engine: Path


# Legacy keys folded into a load case by the migration shim. Older configs put the
# deck stem + displacement node on ``model`` and the limits in a top-level
# ``constraints`` block; those now live per load case. Kept here so both the shim
# and :func:`unknown_keys` agree on what counts as recognised-legacy.
_LEGACY_MODEL_KEYS = ("stem", "disp_node_id")
_LEGACY_TOP_SECTIONS = ("constraints",)


def _migrated_load_cases(data: dict) -> list:
    """Raw ``load_cases`` rows, migrating a legacy single-case config.

    When ``load_cases`` is absent/empty but a legacy ``model.stem`` or
    ``constraints`` block is present, synthesise one load case from them so an old
    single-case YAML keeps running and does not silently lose its deck/limits. An
    explicit ``load_cases`` list is returned unchanged (any legacy
    ``model``/``constraints`` keys are then ignored).
    """
    rows = data.get("load_cases") or []
    if rows:
        return rows
    m = data.get("model") or {}
    c = data.get("constraints") or {}
    if not (m.get("stem") or c):
        return []
    return [{
        "name": "default",
        "stem": m.get("stem", ""),
        "weight": 1.0,
        "disp_node_id": m.get("disp_node_id"),
        "sigma_allow": c.get("sigma_allow"),
        "d_allow": c.get("d_allow"),
    }]


@dataclass
class Config:
    or_paths: ORPaths = field(default_factory=ORPaths)
    run: RunOpts = field(default_factory=RunOpts)
    docker: DockerOpts = field(default_factory=DockerOpts)
    model: Model = field(default_factory=Model)
    beso: Beso = field(default_factory=Beso)
    levelset: LevelSet = field(default_factory=LevelSet)
    tobs: TobsOpts = field(default_factory=TobsOpts)
    hca: HcaOpts = field(default_factory=HcaOpts)
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
    # element removal), "levelset" (nodal level-set, smoother boundaries), "tobs"
    # (binary ILP flips, Sivapuram & Picelli 2018) or "hca" (hybrid cellular
    # automata, LS-TaSC-style density controller). The active block's shared knobs
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
            beso=build(Beso, data.get("beso")),
            levelset=build(LevelSet, data.get("levelset")),
            tobs=build(TobsOpts, data.get("tobs")),
            hca=build(HcaOpts, data.get("hca")),
            manufacturing=build(ManufacturingOpts, data.get("manufacturing")),
            d3plot=build(D3plotOpts, data.get("d3plot")),
            smooth=build(SmoothOpts, data.get("smooth")),
            report=build(ReportOpts, data.get("report")),
            animate=build(AnimateOpts, data.get("animate")),
            load_cases=[build(LoadCase, lc) for lc in _migrated_load_cases(data)],
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
        """Normalised optimiser selector: ``"beso"``, ``"levelset"``, ``"tobs"``
        or ``"hca"``."""
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
        if name == "hca":
            return self.hca
        return self.beso

    def load_case_list(self) -> list[ResolvedCase]:
        """Resolve :attr:`load_cases` into concrete cases with deck paths.

        The load cases are the single source of truth — there is no synthesised
        default. Returns ``[]`` when none are defined (validation requires at
        least one before a run starts; a legacy single-case YAML is migrated into
        one load case on read, see :func:`_migrated_load_cases`).
        """
        case_dir = Path(self.model.case_dir).resolve()
        out: list[ResolvedCase] = []
        for lc in self.load_cases:
            stem = lc.stem
            out.append(ResolvedCase(
                name=lc.name or stem,
                stem=stem,
                weight=float(lc.weight),
                disp_node_id=lc.disp_node_id,
                sigma_allow=lc.sigma_allow,
                d_allow=lc.d_allow,
                starter=case_dir / f"{stem}_0000.rad",
                engine=case_dir / f"{stem}_0001.rad"))
        return out

    def primary_case(self) -> ResolvedCase:
        """The first resolved load case (the deck whose mesh anchors the run).

        Raises if no load cases are defined — callers past validation always have
        at least one."""
        cases = self.load_case_list()
        if not cases:
            raise ValueError("no load cases defined (need at least one)")
        return cases[0]

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

    top_known = (set(sections) | {"load_cases", "optimizer", "work_dir"}
                 | set(_LEGACY_TOP_SECTIONS))    # constraints: migrated, not flagged
    out.extend(k for k in data if k not in top_known)

    for name, klass in sections.items():
        sub = data.get(name)
        if isinstance(sub, dict):
            known = _field_names(klass)
            if name == "model":
                known = known | set(_LEGACY_MODEL_KEYS)   # stem/disp_node_id: migrated
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
    mdl = data.get("model")
    if isinstance(mdl, dict):
        _list_of("model.growth_boxes", mdl.get("growth_boxes"), GrowthBox)

    return out

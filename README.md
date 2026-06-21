# oropt — OpenRadioss-coupled topology optimisation (BESO · level-set · TOBS)

[![CI](https://github.com/pmquang87/structural_optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/pmquang87/structural_optimization/actions/workflows/ci.yml)

Lightweight structural **topology optimisation** that drives the real
**OpenRadioss** implicit nonlinear model in the loop. Its default optimiser is
**BESO** (Bi-directional Evolutionary Structural Optimisation): each iteration
solves the deck, ranks elements by the internal-energy density OpenRadioss already
writes to `/ANIM/ELEM/ENER`, **deletes** the least-important ones (with
bi-directional add-back), and re-solves — removing material while the
high-fidelity solver still reports peak von-Mises stress and a chosen node's
displacement within limits.

Three discrete optimisers plug into this same solve → delete → re-solve loop and
are picked with a single `optimizer:` key — **BESO** (default), a nodal
**level-set** (smoother boundaries), and **TOBS** (integer-linear-programming
binary flips) — all reusing the `/ANIM/ELEM/ENER` energy as their sensitivity.

Built for the AlSi10Mg additively-manufactured elevator linkage (575k TET4,
6 kN pull through rigid cylinders via contact, implicit nonlinear quasi-static).

## Why element deletion (and not SIMP)

OpenRadioss exposes no design sensitivities, and at ~13 min/solve finite-
difference gradients are hopeless. But the per-element energy in `/ANIM/ELEM/ENER`
*is* the BESO sensitivity number, so no adjoint, surrogate, or per-element
material interpolation is needed. Deleting elements rather than softening them
also keeps the implicit stiffness well-conditioned, and:

* **Contacts need no edits.** The linkage contact master is `/SURF/PART/EXT`,
  which OpenRadioss regenerates from the surviving elements; the rigid-cylinder
  masters and the slave node-groups are untouched.
* **The deck is edited verbatim.** Deleted element cards are simply omitted from
  the already-converted `_0000.rad` — no re-conversion, no reformatting.
* **Freed interior nodes are pinned.** Nodes an element-less after deletion would
  make the implicit tangent singular, so a `/GRNOD/NODE` + `/BCS` fixing them is
  injected before `/END` (the converter's free-node guard, generalised).

A 2026 research spike re-examined density-based **SIMP** for this model and still
lands on **no-go**: the compliance sensitivity *is* recoverable from
`/ANIM/ELEM/ENER`, but OpenRadioss has no per-element modulus (so the deck rewrite
turns invasive and breaks `/SURF/PART/EXT` contact regeneration) and the
nonlinear plastic-contact solve mismatches SIMP's linear-elastic theory — full
argument and an offline OC prototype in [`docs/simp_spike.md`](docs/simp_spike.md).

## Architecture

```
oropt/
  config.py   YAML-backed Config (OR paths, run/docker opts, model, constraints, per-optimiser knobs, load cases)
  runner.py   run starter + engine (native np=1, or the Docker MUMPS backend); termination checks
  results.py  anim_to_vtk -> pyvista: per-element energy & von-Mises, loaded-node displacement
  deck.py     parse /NODE + /TETRA4 once; verbatim filtered re-write; free-node pinning; engine trim
  mesh.py     centroids, volumes, sensitivity-filter matrix, connectivity, protected/keep-out regions
  beso.py     sensitivity -> filter + history average -> volume-target threshold + add-back + connectivity
  levelset.py nodal level-set alternative: energy -> nodal velocity -> phi evolution + smoothing -> bisected threshold
  tobs.py     binary-ILP alternative: per-iteration element flips chosen by an integer linear program (scipy HiGHS)
  simp.py     EXPERIMENTAL SIMP/OC prototype — offline maths only, not wired into the loop (see docs/simp_spike.md)
  manufacturing.py additive-manufacturing constraints on the alive mask: min member size (open), symmetry, overhang
  smoothing.py / d3plot.py  post-run: smoothed-surface (STL/VTP) export; OpenRadioss anim -> LS-Dyna d3plot
  report.py   post-run: auto summary report (report.html/report.md) — charts + final-design render from status/history
  animate.py  post-run: topology_evolution.gif from the per-iteration smoothed surfaces (fixed camera, isolated render)
  status.py   status.json / history.csv / topology_latest.vtu (+ per-iter topology_iterNNNN.vtu) + PID + checkpoint
  loop.py     build_optimizer(cfg) -> solve (every load case) -> extract -> update -> repeat; resumable; feasibility gate
  run.py      CLI entry point
  gui/app.py  Streamlit dashboard (input / load cases / constraints / live monitor) — reads status files only
  gui/cases.py  Streamlit-free helpers: load-case table rows <-> LoadCase config objects
```

## Install

A Python 3.12 virtual environment lives in `.venv` (vtk/scipy/streamlit wheels):

```powershell
.venv\Scripts\python -m pip install -e .[gui]
```

Requires a working OpenRadioss install (default `C:\OpenRadioss`) with Intel
oneAPI MPI — the engine is launched as `mpiexec -np 1 engine_win64_impi.exe`
(the bare engine cannot load its MPI DLLs). Threads default to 6 with
`KMP_BLOCKTIME=0` / `OMP_WAIT_POLICY=PASSIVE` (i9-13900H livelock mitigation).

**No native OpenRadioss?** Set `docker.enabled: true` (see *Configuration
highlights*) to run the solver from the Dockerised MUMPS-implicit build instead —
just Docker Desktop and the loaded image, no Intel oneAPI/MPI, AMD or Intel.

## Usage

Headless:

```powershell
.venv\Scripts\python -m oropt.run --config configs\elevator_linkage.yaml          # start
.venv\Scripts\python -m oropt.run --config configs\elevator_linkage.yaml --resume # resume
```

GUI (configure, launch, and live-monitor; safe to close mid-run):

```powershell
.venv\Scripts\streamlit run oropt\gui\app.py
# or, runnable with PyCharm's green Run button (a plain script that boots Streamlit):
.venv\Scripts\python run_gui.py
```

> A Streamlit app cannot be started with `python oropt/gui/app.py` — use
> `streamlit run` or `run_gui.py`. In PyCharm, point the Run configuration at
> `run_gui.py` and the interpreter at `.venv\Scripts\python.exe`.

The Monitor tab auto-refreshes from the status files on a fixed interval
(default 60 s); adjust it with the **Refresh interval (s)** control in the
sidebar. The **Run / output folder** field on the Input tab is editable; leave it
blank to write into the case directory itself (matching the blank-`work_dir`
default), or type an explicit path to override it.

In the **🧮 Queue** tab each run's folder is shown and **`✏` edits** a pending
entry's config / run folder / resume flag in place. When you enqueue runs whose
folders would collide, the queue automatically gives each its own folder
(`…_2`, `…_3`, …) and launches every run there (via `oropt.run --work-dir`), so
queued runs never overwrite each other's status/results.

## Configuration highlights (`configs/elevator_linkage.yaml`)

* `constraints.sigma_allow`, `constraints.d_allow` — the mass-minimisation limits,
  enforced on OpenRadioss's high-fidelity values each iteration.
* `beso.evolution_rate`, `target_volume_fraction`, `filter_radius`,
  `history_weight`, `sensitivity` (`energy`|`vonmises`|`blend`).
* `optimizer` (default `beso`) — selects the discrete topology optimiser. All
  three share the `/ANIM/ELEM/ENER` energy sensitivity, the element-deletion deck
  path, and the multi-load / AM-constraint / connectivity machinery; only the
  per-iteration update differs:
  * `beso` — the default bi-directional evolutionary scheme (`beso:` knobs).
  * `levelset` — a **nodal level-set** (smoother boundaries than BESO's
    element-by-element removal): energy → nodal velocity → φ evolution +
    smoothing → bisected volume-target threshold. Specifics under `levelset:`:
    `dt`, `smoothing_passes`, `band_width`.
  * `tobs` — **TOBS** (Topology Optimisation of Binary Structures): each
    iteration's element flips are chosen by an integer linear program
    (`scipy.optimize.milp` / HiGHS) with a formal move limit and a linearised,
    ε-relaxed volume constraint, instead of a heuristic threshold. Specifics
    under `tobs:`: `flip_limit` (β, max fraction of elements flipped per step)
    and `constraint_relaxation` (ε volume band).

  Each optimiser's `<name>:` block also mirrors the shared knobs
  (`target_volume_fraction`, `evolution_rate`, `filter_radius`, `history_weight`,
  `max_iter`, `convergence_*`, `protect_*`, `archive_*`). Selectable on the GUI.
* **Additive-manufacturing constraints** (`manufacturing:`, all OFF by default) —
  applied to the alive mask each iteration after the optimiser update, for parts
  printed by powder-bed fusion (e.g. AlSi10Mg). `min_member_layers` (morphological
  open removing thin features; 0 = off), `symmetry_planes` (list of
  `{axis: x|y|z, offset: <coord>}`, mirrored *either-alive ⇒ both-alive*), and
  overhang self-support via `build_direction` (`[x,y,z]`, `null` = off) +
  `max_overhang_angle` (cone half-angle in degrees from the build direction).
* **Multiple load cases** (`load_cases:`, empty by default) — optimise one
  structure against several loads (the elevator linkage pulled in different
  directions) by minimising a **weighted-sum compliance**. Each entry is a
  separate deck pair sharing the same mesh, differing only in its applied-load
  cards:

  ```yaml
  load_cases:
    - {name: pull_z, stem: implicit_pull_z, weight: 1.0}
    - {name: pull_x, stem: implicit_pull_x, weight: 0.5, sigma_allow: 480.0}
    - {name: side,   stem: implicit_side,   weight: 0.5, disp_node_id: 10021400}
  ```

  Blank per-case fields inherit the single-case defaults — `stem` → `model.stem`,
  `disp_node_id` → `model.disp_node_id`, `sigma_allow`/`d_allow` → `constraints`.
  All cases must share the same design-part element ids (only the load differs).
  Leave `load_cases` empty for the classic single-solve run (byte-identical
  behaviour). Editable on the GUI's dedicated **Load cases** tab (add/remove rows;
  blank optional cells inherit defaults); the *Monitor* tab then flags that
  σ_max/disp are the worst across all cases. See
  **[How multiple load cases work](#how-multiple-load-cases-work)** below for the
  per-iteration solve → combine → update flow.
* **Keep-out / non-design regions** — `model.freeze_group_ids` (e.g. `[99999999]`,
  any `/GRNOD/NODE` set in the deck) and `model.freeze_node_ids`: every design
  element touching those nodes is frozen and never deleted. Boundary-condition,
  symmetry and contact regions are protected automatically. Frozen elements are
  **excluded from the removal ranking** (they always count as present), so the
  optimiser only ever removes the remaining design material — note that an
  over-large keep-out caps how much mass can be removed (if it already exceeds
  `target_volume_fraction`, no removal is possible).
* `beso.protect_bc_nodes` (default `true`) — whether elements touching the BC
  node-group (`model.bc_group_id`) are frozen. Set it `false` to **allow the
  optimiser to delete material at the BC nodes** too; those nodes stay fixed via
  their own `/BCS` (so the solve is still well-posed) and continue to anchor
  connectivity, so floating islands are still dropped. Exposed as **Allow
  deleting elements at BC nodes** on the GUI's *Constraints / BC* tab.
* `work_dir` — the run/output folder for scratch, checkpoints and status files.
  **Leave it blank to default to the input deck folder (`model.case_dir`)
  itself**, so a run writes its artefacts right next to the deck it optimises;
  set an explicit path (e.g. `runs/run01`) to put outputs elsewhere, or pass
  `oropt.run --work-dir <dir>` to override it for one run (what the queue uses to
  give colliding runs their own folder). The mutated deck always lives in
  the `solve/` sub-folder (`<run_folder>/solve/<stem>_0000.rad`), so the source
  decks in `model.case_dir` are never overwritten.
* `beso.archive_iterations` / `beso.archive_restart` (both default `true`) — see *Outputs* below.
* `d3plot` — post-run conversion of the final OpenRadioss animation into an
  LS-Dyna `d3plot` (viewable in LS-PrePost etc.), **on by default**
  (`d3plot.enabled: true`); one d3plot is produced per load case.
  `tool_root` points at the [Vortex-Radioss](https://github.com/Vortex-CAE/Vortex-Radioss)
  `openradioss_tools` checkout (the folder holding the `vortex_radioss` package).
  The converter runs in an **isolated subprocess** using `python_exe` — blank
  picks `tool_root/.venv` (where lasso-python/tqdm live), so oropt's own
  environment stays clean. It is best-effort: a missing tool, interpreter or
  dependency is logged and skipped, never failing the run. Also exposed as
  **Post-processing — d3plot** on the GUI's *Constraints / BC* tab.
* `smooth` — surface smoothing of the optimised geometry, **on by default**
  (`smooth.enabled: true`). Extracts the design surface, smooths it
  (`method: taubin` volume-preserving, or `laplacian`; `iterations` passes) and
  writes `topology_smoothed.<ext>` (`output_format: stl|vtp|both`) to the run
  folder — a clean deliverable for CAD / 3D-print / review. **Every** per-iteration
  snapshot is smoothed too, into `topology_smoothed_iterNNNN.<ext>`, so the
  smoothed shape evolution is reviewable, not just the final design. Best-effort.
  Exposed under **Post-processing — Surface smoothing** in the GUI.
* `report` — automatic post-run **summary report** (`report.enabled: true` by
  default — it's cheap and read-only). On finish, oropt summarises the run from
  the `status.json`/`history.csv` it already wrote into `report.html` (a
  self-contained page with the convergence charts and a render of the final
  design embedded) and `report.md`: optimiser, start→final volume fraction and %
  mass removed, final σ_max/displacement vs their limits, feasibility,
  iteration count and total wall time. Charts need matplotlib and the final-design
  render an off-screen pyvista (run in an **isolated subprocess**, so even a hard
  GL/driver crash on a headless box can't abort the run); both are best-effort —
  if either is unavailable the report still writes and links the files instead.
  Set `report.charts: false` or `report.render_topology: false` to skip those
  pieces.
* `animate` — automatic post-run **topology-evolution GIF** (`animate.enabled:
  true` by default). On finish, oropt renders the per-iteration *smoothed*
  surfaces (`topology_smoothed_iterNNNN.<ext>`, falling back to the raw
  `topology_iterNNNN.vtu` snapshots when `smooth` is off) from a **single fixed
  camera** — framed once on the union of all snapshots' bounds, so the part loses
  material *in place* instead of rescaling — and assembles them into
  `topology_evolution.gif` in the run folder. Frames are drawn by an **isolated
  off-screen pyvista subprocess** (like the report's render, so a hard GL/driver
  crash on a headless box can't abort the run) and encoded with Pillow.
  Best-effort: a run with fewer than two snapshots, or a missing/failing
  dependency, is logged and skipped. The camera angle is `animate.view` — a
  built-in preset (`iso` / `front` / `back` / `left` / `right` / `top` /
  `bottom`) **or the name of a user-defined angle** — nudged by `azimuth` /
  `elevation` (degrees), so any viewpoint is reachable and it stays fixed across
  the clip. Define reusable named angles under `animate.custom_views` (each a
  `base` preset + `azimuth`/`elevation` offsets); they become selectable by name
  (and appear in the GUI's *Camera angle* dropdown):
  ```yaml
  animate:
    view: three_quarter            # built-in preset OR a custom name below
    custom_views:
      - {name: three_quarter, base: front, azimuth: 40, elevation: 25}
  ```
  Other tunables: `fps`, `color`, `opacity` (0..1 — drop below 1 to make the
  design see-through so internal structure shows; transparency uses depth peeling
  when the driver supports it), `show_labels`, `hold_last`, `window_w`/`window_h`.
  The whole **Evolution animation** block — enable, custom angles, camera angle,
  azimuth/elevation, fps, opacity, labels — is editable under *Post-processing* in
  the GUI. Re-buildable for any existing run folder without re-optimising via
  `python -m oropt.animate <run_dir>` (e.g.
  `--view top --fps 8 --color orange --opacity 0.5`).
* `docker` — optionally run the solver via the **Dockerised OpenRadioss MUMPS
  build** instead of the native Windows binaries (no Intel oneAPI/MKL/MPI; works
  on AMD or Intel). Set `docker.enabled: true` with the loaded `image`
  (`openradioss-mumps:20260520`) and `np`/`nt` (the container supports real MPI,
  so `np > 1` is fine — keep `np × nt` ≤ cores). The run folder is bind-mounted
  to `/data` and the container writes its `.out`/animation/`T01`/`.rst` back
  there, so the rest of the pipeline is unchanged. Requires Docker Desktop
  running; selectable as **Solver backend** on the GUI's *Input* tab.

## How multiple load cases work

Multiple load cases optimise **one** shared structure against several loads at
once. The primary (first) case defines the geometry, mesh and protected set; every
other case is a separate deck pair (`<stem>_0000.rad` / `<stem>_0001.rad`) that
**must share the same design-part element ids** — only its load cards differ. There
is a single `alive` element mask and a single optimiser; the loads are all that
vary.

Each iteration runs the same four steps:

1. **Solve every case.** The current `alive` design is written into each case's
   deck and solved **sequentially**, each in its own `solve/case_<i>/` directory
   (so decks, listings and animations never collide). Runtime is therefore ≈ N× a
   single-case iteration. If any case fails to solve, the iteration aborts.
2. **Combine into one decision.** The per-case results are fused two ways:
   * **Objective (sensitivity)** — a *per-case-normalised weighted sum*
     `s_e = Σ_i wᵢ·(rawᵢ_e / max rawᵢ)`. Normalising each case by its own peak
     before weighting makes the weights express *relative* importance regardless
     of how the cases' absolute strain-energy magnitudes differ.
   * **Constraints (feasibility)** — *worst-case*: the reported `sigma_max` /
     `disp` are the maxima across cases, and the design is **feasible only when
     every case satisfies its own `sigma_allow` / `d_allow`**.
3. **One shared design update.** From here the loop is identical to a single-load
   run — it sees only the combined sensitivity and the worst-case feasibility:
   spatial filter + history blend, convergence check, then the target-volume /
   `alive`-mask update. This combining sits in the loop **above** the optimiser, so
   multiple load cases work with **any** optimiser (`beso`, `levelset`, `tobs`)
   unchanged.
4. **Next iteration** re-solves *all* cases against the new design, and the cycle
   repeats until convergence or `max_iter`.

So after the first iteration of all cases is computed, the N per-case results
collapse into a single sensitivity field and a single feasibility verdict, which
drive one new global `alive` mask — the next iteration then re-solves every case on
that mask. Cost scales roughly linearly with the number of cases (they run
sequentially, not in parallel).

Post-processing covers **every** case: the per-iteration archive
(`iter_NNNN/<stem>…`) and the final-design d3plot (`d3plot/<stem>.d3plot`) are
written per case, while surface smoothing emits the one shared design (the final
`topology_smoothed.<ext>` plus each `topology_smoothed_iterNNNN.<ext>`). Leave
`load_cases` empty for the classic single-solve run — the multi-case path then
collapses to exactly the original single-solve behaviour (byte-identical).

## Outputs

Every iteration the loop writes, into the run folder (`work_dir`, or `case_dir` when blank):

* `status.json` / `history.csv` — live scalar state + one row per iteration.
* `topology_latest.vtu` — the current alive mesh (overwritten), for the GUI.
* `topology_iterNNNN.vtu` — an **immutable per-iteration snapshot** of the alive
  mesh (sensitivity + von-Mises fields), so the topology evolution can be
  replayed/animated after the run. These are small (only the surviving tets).

**`beso.archive_iterations`** (on by default) keeps each iteration's key
OpenRadioss outputs under `work_dir/iter_NNNN/` before the `solve/` folder is
recycled for the next iteration: the mutated `<stem>_0000.rad`, the final
animation state(s) `<stem>A0*`, and the engine listing `<stem>_0001.out`.
**`beso.archive_restart`** (**off by default**) additionally copies the restart
(`<stem>*.rst`), preserving the *full* solver state of every iteration for
replay/debug. With multiple load cases **every** case is archived under
`iter_NNNN/`, each case in its own stem-named sub-folder
(`iter_NNNN/<stem>/<stem>_0000.rad`, …) so its files stay grouped; a single-case
run archives straight into `iter_NNNN/` (byte-identical to before). Set either
flag to `false` to save disk (see the note below).

With **`d3plot.enabled`** (on by default), once the run finishes each load case's
final animation is converted to an LS-Dyna d3plot and written to
`work_dir/d3plot/<stem>.d3plot` (+ its `.d3plotNN` state files) — one per case.
The converter lives outside oropt; point `d3plot.tool_root` at the
`openradioss_tools` checkout (or set the `OROPT_VORTEX_ROOT` environment variable
and leave `tool_root` blank). A missing tool just skips the conversion.
With **`smooth.enabled`** (on by default), the design surface is extracted,
smoothed and written to `work_dir/topology_smoothed.<ext>` (STL/VTP), and every
per-iteration snapshot likewise into `work_dir/topology_smoothed_iterNNNN.<ext>`.

Unless **`report.enabled: false`**, the run also writes a summary report —
`work_dir/report.html` (self-contained: convergence charts + a final-design
render embedded) and `work_dir/report.md`, alongside the `report_*.png` charts —
recapping optimiser, start→final volume fraction and % mass removed, final
σ_max/displacement vs limits, feasibility, iteration count and total wall time.

> **Disk cost.** Archiving is **on by default** and adds up fast: tens of MB per
> iteration *per load case* (deck + animation), **plus** the ~345 MB restart
> (`<stem>_0000_0001.rst`) per iteration per case while `archive_restart` is on
> (also the default) — the restart alone is ~50 GB over a long single-case run and
> scales with the number of load cases. Per-iteration smoothed surfaces (on by
> default) add a little more. Set `beso.archive_restart: false` and/or
> `beso.archive_iterations: false` to trim this when you don't need every
> iteration's full solver state.

## Honest caveats

* **BESO is heuristic** — sensitive to evolution rate, filter radius and history
  weight; start conservative and watch the mass / σ / displacement traces.
* **Cost** — ~13 min/solve × 50–150 iterations ≈ 11–33 h. Per-iteration
  checkpoints make runs resumable; develop on a coarse proxy mesh.
* **np = 1 on the native backend** — SPMD implicit + solid contact segfaults
  (documented upstream limitation) on the Intel/Windows build, so the native path
  has no domain parallelism. The Docker MUMPS backend (`docker.enabled`) does
  support real MPI (`np > 1`).
* **Self-contact** (`/INTER/TYPE7/90001`) may see newly-exposed cavity faces after
  deletion; usually harmless (interior cavities contact nothing) but worth a look
  after the first deletions.
* The full nonlinear model (LAW36 plasticity + contact) is solved every
  iteration — there is no linear-elastic simplification.

## Validation status

* Runner + extraction reproduce the known-good baseline exactly
  (σ_max = 308.305 MPa, disp = 1.229 mm, NORMAL TERMINATION).
* A hand-deletion (−16 % volume, 16 352 freed nodes auto-pinned) produces a deck
  that OpenRadioss solves to NORMAL TERMINATION.
* `pytest` (124 tests, all hermetic — no OpenRadioss needed) covers deck
  round-trip/pinning, mesh geometry/connectivity/protection, BESO ranking/threshold,
  the **level-set** (bisected volume targeting / protected / φ-thresholding /
  connectivity) and **TOBS** (ILP feasibility / move-limit / volume targeting /
  protected) updates and optimiser selection, the **multi-load** weighted-sum and
  worst-case feasibility aggregation, the **additive-manufacturing** constraints,
  the **Docker** command construction, **d3plot**/**surface-smoothing** post-processing,
  the offline **SIMP** OC/bisection/projection prototype, status/checkpoint
  round-trips, VTK extraction, per-iteration snapshot/archive file-writing, and the
  run-folder fallback + source-deck isolation.

This project consumes decks produced by the sibling `k_to_rad_converter`
(LS-DYNA → OpenRadioss); see that project for the conversion step.
```

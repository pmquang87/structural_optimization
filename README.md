# oropt — OpenRadioss-coupled topology optimisation (BESO · level-set · TOBS · HCA)

[![CI](https://github.com/pmquang87/structural_optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/pmquang87/structural_optimization/actions/workflows/ci.yml)

Lightweight structural **topology optimisation** that drives the real
**OpenRadioss** implicit nonlinear model in the loop. Its default optimiser is
**BESO** (Bi-directional Evolutionary Structural Optimisation): each iteration
solves the deck, ranks elements by the internal-energy density OpenRadioss already
writes to `/ANIM/ELEM/ENER`, **deletes** the least-important ones (with
bi-directional add-back), and re-solves — removing material while the
high-fidelity solver still reports peak von-Mises stress and a chosen node's
displacement within limits.

Four discrete optimisers plug into this same solve → delete → re-solve loop and
are picked with a single `optimizer:` key — **BESO** (default), a nodal
**level-set** (smoother boundaries), **TOBS** (integer-linear-programming
binary flips), and **HCA** (hybrid cellular automata, the LS-TaSC method) — all
reusing the `/ANIM/ELEM/ENER` energy as their sensitivity.

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
  mesh.py     centroids, volumes, sensitivity-filter matrix, connectivity, protected/keep-out regions, growth-box selection
  beso.py     sensitivity -> filter + history average -> volume-target threshold + add-back + connectivity
  levelset.py nodal level-set alternative: energy -> nodal velocity -> phi evolution + smoothing -> bisected threshold
  tobs.py     binary-ILP alternative: per-iteration element flips chosen by an integer linear program (scipy HiGHS)
  hca.py      hybrid-cellular-automata alternative (LS-TaSC-style): per-element virtual density driven by a setpoint controller
  simp.py     EXPERIMENTAL SIMP/OC prototype — offline maths only, not wired into the loop (see docs/simp_spike.md)
  manufacturing.py additive-manufacturing constraints on the alive mask: min member size (open), symmetry, overhang
  smoothing.py / d3plot.py  post-run: smoothed-surface (STL/VTP) export; OpenRadioss anim -> LS-Dyna d3plot
  report.py   post-run: auto summary report (report.html/report.md) — charts + interactive/static final-design view from status/history
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

Optional extras: `report3d` adds the **trame** export backend
(`pip install -e .[gui,report3d]`) so `report.html` embeds an interactive
zoom/rotate viewer of the final design (via pyvista's `export_html`); without it
the report falls back to a static image. `dev` adds pytest/ruff.

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

Adding a run to the queue **freezes a snapshot** of the config at that moment
(saved in the model's case directory under `queue_configs/`) and the queued run
launches from that copy — so editing the working config afterwards never changes
a run already queued. The
**🧮 Queue** tab shows each entry's snapshot and folder and lets you reorder
(`⬆`/`⬇`) or remove (`✖`) pending entries; there is no in-place edit, since the
snapshot *is* the run. When you enqueue runs whose folders would collide, the
queue automatically gives each its own folder (`…_2`, `…_3`, …) and launches
every run there (via `oropt.run --work-dir`), so queued runs never overwrite each
other's status/results.

The sidebar **Run state** and the Monitor tab follow whatever run is actually
live — the selected config's, or a queued run in its own reserved folder — so
they stay in sync with the queue instead of showing idle while a queued run is
in progress.

## Configuration highlights (`configs/elevator_linkage.yaml`)

* Feasibility limits `sigma_allow` / `d_allow` (enforced on OpenRadioss's
  high-fidelity values each iteration) are defined **per load case** in
  `load_cases:` — see *Multiple load cases* below. There is no global
  `constraints` block; even a single-load run is one load case.
* `beso.evolution_rate`, `target_volume_fraction`, `filter_radius`,
  `history_weight`, `sensitivity` (`energy`|`vonmises`|`blend`).
* `optimizer` (default `beso`) — selects the discrete topology optimiser. All
  four share the `/ANIM/ELEM/ENER` energy sensitivity, the element-deletion deck
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
  * `hca` — **HCA** (Hybrid Cellular Automata; Tovar et al. 2006, the method
    behind LS-TaSC, designed for exactly this gradient-free nonlinear/contact
    regime): every element keeps a persistent *virtual density* `x_e ∈ [0.01, 1]`
    that a proportional controller drives toward a uniform energy-density
    setpoint `S*` (`x_e += kp·(S_e − S*)/S*`, move-limited), the setpoint found
    by bisection so the thresholded design (alive iff `x_e ≥ 0.5`) hits the
    per-iteration volume target; the spatial filter doubles as the cellular
    automaton's neighbourhood averaging. Specifics under `hca:`: `kp`
    (controller gain), `move_limit` (max density change per iteration; keep
    `min(kp, move_limit) > 0.5` so removal can track the target step-for-step,
    lower values give damped multi-iteration decay), `field_history_weight`
    (extra HCA-internal blend of the energy field with previous iterations,
    LS-TaSC's multi-iteration weighted sum; `1.0` = off).

  Each optimiser's `<name>:` block also mirrors the shared knobs
  (`target_volume_fraction`, `evolution_rate`, `filter_radius`, `history_weight`,
  `max_iter`, `convergence_*`, `protect_*`, `archive_*`, and the feasibility
  back-off controller below). Selectable on the GUI.
* **Feasibility back-off controller** (`backoff_gain` / `backoff_cap` /
  `damping_threshold`, mirrored on each optimiser block; defaults reproduce the
  classic binary gate exactly) — how the per-iteration volume target reacts to
  the constraint *values*. By default any violated limit grows the target by one
  full `evolution_rate` step and any feasible design shrinks it by one — an
  on/off gate that tends to ping-pong across the limit. With `backoff_gain > 0`
  the growth step becomes proportional to the worst constraint-utilisation
  ratio `v = max(σ_max/σ_allow, d/d_allow)` over the load cases,
  `ER·min(gain·(v−1), cap)`, the way TOSCA's controller mode / LS-TaSC's
  constrained scaling react to the stress level rather than a flag (size the
  gain so `gain·(typical overshoot) ≈ 1`, e.g. 10–20). With
  `damping_threshold < 1` (0.9–0.95 typical) removal slows by
  `(1−v)/(1−threshold)` once `v` exceeds the threshold, so the design glides
  into the limit instead of overshooting and oscillating feasible/infeasible.
  `addback_stress_bias > 0` additionally makes the *add-back stress-responsive*:
  whenever a stress limit is violated, the sensitivity driving that update is
  scaled by `(1 + bias·σ_vm/σ_allow)` (spatially filtered so the overstress
  bleeds into the neighbouring void elements), so the material the back-off
  recovers lands near the overstressed region instead of wherever the energy
  ranking happens to point.
* **Additive-manufacturing constraints** (`manufacturing:`, all OFF by default) —
  applied to the alive mask each iteration after the optimiser update, for parts
  printed by powder-bed fusion (e.g. AlSi10Mg). `min_member_layers` (morphological
  open removing thin features; 0 = off), `symmetry_planes` (list of
  `{axis: x|y|z, offset: <coord>}`, mirrored *either-alive ⇒ both-alive*), and
  overhang self-support via `build_direction` (`[x,y,z]`, `null` = off) +
  `max_overhang_angle` (cone half-angle in degrees from the build direction).
* **Load cases** (`load_cases:`, at least one required) — the single source of
  truth for each deck's `stem`, its constrained `disp_node_id`, and its
  feasibility limits `sigma_allow` / `d_allow`. A single-load run is just **one**
  load case; add more to optimise one structure against several loads (the
  elevator linkage pulled in different directions) by minimising a **weighted-sum
  compliance**. Each entry is a separate deck pair sharing the same mesh,
  differing only in its applied-load cards:

  ```yaml
  load_cases:
    - {name: pull_z, stem: implicit_pull_z, weight: 1.0, disp_node_id: 10021367, sigma_allow: 250.0, d_allow: 1.0}
    - {name: pull_x, stem: implicit_pull_x, weight: 0.5, disp_node_id: 10021367, sigma_allow: 480.0, d_allow: 1.0}
    - {name: side,   stem: implicit_side,   weight: 0.5, disp_node_id: 10021400, sigma_allow: 250.0, d_allow: 2.0}
  ```

  `stem` is **required** on every row; `weight` defaults to 1, and `disp_node_id`,
  `sigma_allow` and `d_allow` may be omitted — a blank `sigma_allow`/`d_allow`
  leaves that quantity **unconstrained** (no feasibility limit), and a blank
  `disp_node_id` tracks no displacement node.
  All cases must share the same design-part element ids (only the load differs).
  Editable on the GUI's dedicated **Load cases** tab (add/remove rows); the
  *Monitor* tab then flags that σ_max/disp are the worst across all cases, with a
  per-case breakdown. A legacy single-case config (old `model.stem` +
  `constraints:` block, no `load_cases`) is migrated into one load case on read.
  See
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
* **Stress-exclusion regions** — `model.stress_exclude_group_ids` (e.g.
  `[999999998]`, any `/GRNOD/NODE` set in the deck) and
  `model.stress_exclude_node_ids`: every design element touching those nodes has
  its von-Mises **ignored** — dropped from the reported peak `σ_max`, the
  feasibility check, and the stress shown in the *Monitor* and `report.html`. Use
  it for a **known hot-spot a later design phase will fix** (e.g. around a small
  cylinder) so that local artefact can't keep the design infeasible or distort the
  loop's back-off. Unlike the keep-out set these elements are **not frozen** — they
  still take part in the optimisation; list them in `freeze_*` as well if you also
  want to protect them from removal. Editable on the GUI's *Optimiser / Output* tab;
  the *Monitor* and report then note how many elements σ_max is ignoring.
* **Growth boxes — add material** (`model.growth_boxes`, none by default) —
  axis-aligned boxes (two opposite corners, like LS-DYNA `*DEFINE_BOX` /
  Radioss `/BOX/RECTA`; multiple boxes act as a union) marking **candidate
  growth material**: every design element whose centroid lies inside a box
  starts the run **void**, and the optimiser's bi-directional update (BESO
  add-back, TOBS `{0,+1}` flips, level-set growth) may *add* it where the load
  path wants — so the design can grow material where the original part had
  none, e.g. a reinforcement rib beyond the original envelope. The box volume
  must be **pre-meshed** into the design part first (same
  `/TETRA4/<design_part_id>` block, node-conformal interface with the part —
  imprint + merge coincident interface nodes — and node ids ≥
  `design_node_min`); run start validates this and errors on an empty box, on
  non-design node ids, and on candidates not node-connected to the structure.
  Candidates are never frozen (a box overlapping a keep-out region stays
  growable, not force-materialised), iteration 0 solves exactly the original
  part, and volume fractions are then relative to the **enlarged**
  (part + boxes) design space. With BESO, keep `max_add_ratio` ≥
  `evolution_rate` so growth isn't throttled below the feasibility back-off
  step (validation warns otherwise). Editable as a table on the GUI's
  *Optimiser / Output* tab; the *Monitor* shows how many candidate elements
  have been grown. Full design study in
  [`docs/add_material_boxes.md`](docs/add_material_boxes.md):

  ```yaml
  model:
    growth_boxes:
      - {name: rib_top,  x_min: 10.0, x_max: 40.0, y_min: -5.0, y_max: 5.0, z_min: 0.0, z_max: 25.0}
      - {name: gusset_l, x_min: -20.0, x_max: 0.0, y_min: -5.0, y_max: 5.0, z_min: 0.0, z_max: 12.0}
  ```
* `beso.protect_bc_nodes` (default `true`) — whether elements touching the BC
  node-group (`model.bc_group_id`) are frozen. Set it `false` to **allow the
  optimiser to delete material at the BC nodes** too; those nodes stay fixed via
  their own `/BCS` (so the solve is still well-posed) and continue to anchor
  connectivity, so floating islands are still dropped. Exposed as **Allow
  deleting elements at BC nodes** on the GUI's *Optimiser / Output* tab.
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
  **Post-processing — d3plot** on the GUI's *Optimiser / Output* tab.
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
  self-contained page with the convergence charts and the final design embedded)
  and `report.md`: optimiser, start→final volume fraction and % mass removed,
  final σ_max/displacement vs their limits, feasibility, iteration count and total
  wall time. The final design is shown as an **interactive zoom/rotate viewer**
  (`report.interactive_topology: true`, the same VTK.js scene as the Monitor tab,
  via pyvista's `export_html`) — this needs the optional **trame** backend (the
  `report3d` extra; see *Install*). Without it, oropt falls back to a static
  off-screen-pyvista **PNG** (`report.render_topology`), and then to a plain file
  link. Charts need matplotlib; every render runs in an **isolated subprocess**, so
  even a hard GL/driver crash on a headless box can't abort the run, and each piece
  is best-effort — anything unavailable is logged and the report still writes. Set
  `report.charts: false`, `report.interactive_topology: false` or
  `report.render_topology: false` to skip those pieces.
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
   multiple load cases work with **any** optimiser (`beso`, `levelset`, `tobs`, `hca`)
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
`topology_smoothed.<ext>` plus each `topology_smoothed_iterNNNN.<ext>`). A single
load case is the classic single-solve run — with one case the multi-case path
collapses to exactly that behaviour.

## Outputs

Every iteration the loop writes, into the run folder (`work_dir`, or `case_dir` when blank):

* `config_used.yaml` — a snapshot of the exact config this run used, written at
  start-up so each result set is reproducible from its own folder.
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
`work_dir/report.html` (self-contained: convergence charts + the final design
embedded) and `work_dir/report.md`, alongside the `report_*.png` charts —
recapping optimiser, start→final volume fraction and % mass removed, final
σ_max/displacement vs limits, feasibility, iteration count and total wall time.
With the optional `report3d` extra installed, the final design is an interactive
zoom/rotate viewer inlined into `report.html` (and also written as the standalone
`work_dir/report_topology.html`); otherwise it's the static `report_topology.png`.

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
* `pytest` (301 tests, all hermetic — no OpenRadioss needed) covers deck
  round-trip/pinning, mesh geometry/connectivity/protection, BESO ranking/threshold,
  the **growth boxes** (candidate selection, run-start guards, growth through
  each optimiser's update),
  the **level-set** (bisected volume targeting / protected / φ-thresholding /
  connectivity), **TOBS** (ILP feasibility / move-limit / volume targeting /
  protected) and **HCA** (setpoint bisection / move-limited density decay /
  candidate growth / protected) updates and optimiser selection, the **multi-load** weighted-sum and
  worst-case feasibility aggregation, the **additive-manufacturing** constraints,
  the **Docker** command construction, **d3plot**/**surface-smoothing** post-processing,
  the offline **SIMP** OC/bisection/projection prototype, status/checkpoint
  round-trips, VTK extraction, per-iteration snapshot/archive file-writing, and the
  run-folder fallback + source-deck isolation.

This project consumes decks produced by the sibling `k_to_rad_converter`
(LS-DYNA → OpenRadioss); see that project for the conversion step.
```

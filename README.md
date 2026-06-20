# oropt — OpenRadioss-coupled topology optimisation (BESO)

Lightweight structural **topology optimisation** that drives the real
**OpenRadioss** implicit nonlinear model in the loop. It implements **BESO**
(Bi-directional Evolutionary Structural Optimisation): each iteration solves the
deck, ranks elements by the internal-energy density OpenRadioss already writes to
`/ANIM/ELEM/ENER`, **deletes** the least-important ones (with bi-directional
add-back), and re-solves — removing material while the high-fidelity solver still
reports peak von-Mises stress and a chosen node's displacement within limits.

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

## Architecture

```
oropt/
  config.py   YAML-backed Config (OR paths, run opts, model, constraints, BESO knobs)
  runner.py   run starter + mpiexec engine (np=1) with the proven env; termination checks
  results.py  anim_to_vtk -> pyvista: per-element energy & von-Mises, loaded-node displacement
  deck.py     parse /NODE + /TETRA4 once; verbatim filtered re-write; free-node pinning; engine trim
  mesh.py     centroids, volumes, sensitivity-filter matrix, connectivity, protected/keep-out regions
  beso.py     sensitivity -> filter + history average -> volume-target threshold + add-back + connectivity
  levelset.py nodal level-set alternative: energy -> nodal velocity -> phi evolution + smoothing -> bisected threshold
  manufacturing.py additive-manufacturing constraints on the alive mask: min member size (open), symmetry, overhang
  status.py   status.json / history.csv / topology_latest.vtu (+ per-iter topology_iterNNNN.vtu) + PID + checkpoint
  loop.py     solve (every load case) -> extract -> rank -> delete -> repeat; resumable; constraint feasibility gate
  run.py      CLI entry point
  gui/app.py  Streamlit dashboard (input + load-case table / constraints / live monitor) — reads status files only
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
blank to use a `work/` sub-folder inside the case directory (matching the
blank-`work_dir` default), or type an explicit path to override it.

## Configuration highlights (`configs/elevator_linkage.yaml`)

* `constraints.sigma_allow`, `constraints.d_allow` — the mass-minimisation limits,
  enforced on OpenRadioss's high-fidelity values each iteration.
* `beso.evolution_rate`, `target_volume_fraction`, `filter_radius`,
  `history_weight`, `sensitivity` (`energy`|`vonmises`|`blend`).
* `optimizer` (default `beso`) — selects the topology optimiser. Set to
  `levelset` to drive the run with the **nodal level-set** optimiser instead
  (smoother boundaries than BESO's element-by-element removal). Its knobs live
  under `levelset:` and mirror the shared BESO ones (`target_volume_fraction`,
  `evolution_rate`, `filter_radius`, `history_weight`, `max_iter`, `convergence_*`,
  `protect_*`, `archive_*`) plus level-set specifics `levelset.dt`,
  `levelset.smoothing_passes`, `levelset.band_width`.
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

  Every iteration solves **all** cases sequentially (each under
  `solve/case_<i>/`, so runtime is N× a single-case run) and extracts per-element
  energy for each. The BESO/level-set sensitivity is the per-case-normalised
  weighted sum `s_e = Σ_i wᵢ·energy_eⁱ` (normalising each case by its own peak
  makes the weights comparable across loads); the design is **feasible only when
  every case is** (status reports the worst-case `sigma_max`/`disp`). Blank
  per-case fields inherit the single-case defaults — `stem` → `model.stem`,
  `disp_node_id` → `model.disp_node_id`, `sigma_allow`/`d_allow` → `constraints`.
  Leave `load_cases` empty for the classic single-solve run (behaviour is
  byte-identical). All cases must share the same design-part element ids (only the
  load differs); the post-run d3plot/smoothing use the primary (first) case.
  Editable as a **Load cases** table on the GUI's *Input* tab (add/remove rows;
  blank optional cells inherit defaults); the *Monitor* tab then flags that
  σ_max/disp are the worst across all cases.
* **Keep-out / non-design regions** — `model.freeze_group_ids` (e.g. `[99999999]`,
  any `/GRNOD/NODE` set in the deck) and `model.freeze_node_ids`: every design
  element touching those nodes is frozen and never deleted. Boundary-condition,
  symmetry and contact regions are protected automatically. Frozen elements are
  **excluded from the removal ranking** (they always count as present), so the
  optimiser only ever removes the remaining design material — note that an
  over-large keep-out caps how much mass can be removed (if it already exceeds
  `target_volume_fraction`, no removal is possible).
  symmetry and contact regions are protected automatically.
* `beso.protect_bc_nodes` (default `true`) — whether elements touching the BC
  node-group (`model.bc_group_id`) are frozen. Set it `false` to **allow the
  optimiser to delete material at the BC nodes** too; those nodes stay fixed via
  their own `/BCS` (so the solve is still well-posed) and continue to anchor
  connectivity, so floating islands are still dropped. Exposed as **Allow
  deleting elements at BC nodes** on the GUI's *Constraints / BC* tab.
* `work_dir` — the run/output folder for scratch, checkpoints and status files.
  **Leave it blank to default to a `work/` sub-folder inside the input deck folder
  (`model.case_dir/work`)**, so a run writes its artefacts right next to the deck
  it optimises without cluttering the source folder; set an explicit path
  (e.g. `runs/run01`) to put outputs elsewhere. The mutated deck always lives in
  the `solve/` sub-folder (`<run_folder>/solve/<stem>_0000.rad`), so the source
  decks in `model.case_dir` are never overwritten.
* `beso.archive_iterations` / `beso.archive_restart` (both default `false`) — see *Outputs* below.
* `d3plot` — optional post-run conversion of the final OpenRadioss animation into
  an LS-Dyna `d3plot` (viewable in LS-PrePost etc.). Set `d3plot.enabled: true`;
  `tool_root` points at the [Vortex-Radioss](https://github.com/Vortex-CAE/Vortex-Radioss)
  `openradioss_tools` checkout (the folder holding the `vortex_radioss` package).
  The converter runs in an **isolated subprocess** using `python_exe` — blank
  picks `tool_root/.venv` (where lasso-python/tqdm live), so oropt's own
  environment stays clean. It is best-effort: a missing tool, interpreter or
  dependency is logged and skipped, never failing the run. Also exposed as
  **Post-processing — d3plot** on the GUI's *Constraints / BC* tab.
* `smooth` — optional surface smoothing of the **final optimised geometry**. Set
  `smooth.enabled: true` to extract the final design's surface, smooth it
  (`method: taubin` volume-preserving, or `laplacian`; `iterations` passes) and
  write `topology_smoothed.<ext>` (`output_format: stl|vtp|both`) to the run
  folder — a clean deliverable for CAD / 3D-print / review. Best-effort. Exposed
  under **Post-processing — Surface smoothing** in the GUI.
* `docker` — optionally run the solver via the **Dockerised OpenRadioss MUMPS
  build** instead of the native Windows binaries (no Intel oneAPI/MKL/MPI; works
  on AMD or Intel). Set `docker.enabled: true` with the loaded `image`
  (`openradioss-mumps:20260520`) and `np`/`nt` (the container supports real MPI,
  so `np > 1` is fine — keep `np × nt` ≤ cores). The run folder is bind-mounted
  to `/data` and the container writes its `.out`/animation/`T01`/`.rst` back
  there, so the rest of the pipeline is unchanged. Requires Docker Desktop
  running; selectable as **Solver backend** on the GUI's *Input* tab.

## Outputs

Every iteration the loop writes, into the run folder (`work_dir`, or `case_dir/work`):

* `status.json` / `history.csv` — live scalar state + one row per iteration.
* `topology_latest.vtu` — the current alive mesh (overwritten), for the GUI.
* `topology_iterNNNN.vtu` — an **immutable per-iteration snapshot** of the alive
  mesh (sensitivity + von-Mises fields), so the topology evolution can be
  replayed/animated after the run. These are small (only the surviving tets).

Set **`beso.archive_iterations: true`** to also keep each iteration's key
OpenRadioss outputs under `work_dir/iter_NNNN/` before the `solve/` folder is
recycled for the next iteration: the mutated `<stem>_0000.rad`, the final
animation state(s) `<stem>A0*`, and the engine listing `<stem>_0001.out`. Add
**`beso.archive_restart: true`** to also copy the restart (`<stem>*.rst`),
preserving the *full* solver state of every iteration for replay/debug.

When **`d3plot.enabled: true`**, once the run finishes the final design's
animation is converted to an LS-Dyna d3plot and written to
`work_dir/d3plot/<stem>.d3plot` (+ its `.d3plotNN` state files). When
**`smooth.enabled: true`**, the final design's surface is extracted, smoothed and
written to `work_dir/topology_smoothed.<ext>` (STL/VTP).

> **Disk cost.** Archiving is off by default because it adds up: tens of MB per
> iteration (deck + animation), so a 50–150 iteration run can reach several GB.
> The ~345 MB restart (`<stem>_0000_0001.rst`) is excluded unless you opt in with
> `archive_restart` — that alone is ~50 GB over a long run, so enable it only when
> you truly need every iteration's full state.

## Honest caveats

* **BESO is heuristic** — sensitive to evolution rate, filter radius and history
  weight; start conservative and watch the mass / σ / displacement traces.
* **Cost** — ~13 min/solve × 50–150 iterations ≈ 11–33 h. Per-iteration
  checkpoints make runs resumable; develop on a coarse proxy mesh.
* **np = 1 only** — SPMD implicit + solid contact segfaults (documented upstream
  limitation), so there is no domain parallelism.
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
* `pytest` covers deck round-trip/pinning, mesh geometry/connectivity/protection,
  BESO ranking/threshold, the level-set bisected volume targeting / protected /
  phi-thresholding self-consistency / connectivity / optimiser selection,
  status/checkpoint round-trips, VTK extraction, per-iteration snapshot/archive
  file-writing, and the run-folder fallback + source-deck isolation.

This project consumes decks produced by the sibling `k_to_rad_converter`
(LS-DYNA → OpenRadioss); see that project for the conversion step.
```

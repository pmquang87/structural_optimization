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
  status.py   status.json / history.csv / topology_latest.vtu (+ per-iter topology_iterNNNN.vtu) + PID + checkpoint
  loop.py     solve -> extract -> rank -> delete -> repeat; resumable; constraint feasibility gate
  run.py      CLI entry point
  gui/app.py  Streamlit dashboard (input / constraints / live monitor) — reads status files only
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
sidebar. The **Run / output folder** field on the Input tab defaults to the case
directory (matching the blank-`work_dir` default).

## Configuration highlights (`configs/elevator_linkage.yaml`)

* `constraints.sigma_allow`, `constraints.d_allow` — the mass-minimisation limits,
  enforced on OpenRadioss's high-fidelity values each iteration.
* `beso.evolution_rate`, `target_volume_fraction`, `filter_radius`,
  `history_weight`, `sensitivity` (`energy`|`vonmises`|`blend`).
* **Keep-out / non-design regions** — `model.freeze_group_ids` (e.g. `[99999999]`,
  any `/GRNOD/NODE` set in the deck) and `model.freeze_node_ids`: every design
  element touching those nodes is frozen and never deleted. Boundary-condition,
  symmetry and contact regions are protected automatically.
* `work_dir` — the run/output folder for scratch, checkpoints and status files.
  **Leave it blank to default to the input deck folder (`model.case_dir`)**, so a
  run writes its artefacts next to the deck it optimises; set an explicit path
  (e.g. `runs/run01`) to keep outputs separate. The mutated deck always lives in
  the `solve/` sub-folder (`<run_folder>/solve/<stem>_0000.rad`), so the source
  `<run_folder>/<stem>_0000.rad` is never overwritten — even when the run folder
  *is* the input folder.
* `beso.archive_iterations` (default `false`) — see *Outputs* below.

## Outputs

Every iteration the loop writes, into the run folder (`work_dir`, or `case_dir`):

* `status.json` / `history.csv` — live scalar state + one row per iteration.
* `topology_latest.vtu` — the current alive mesh (overwritten), for the GUI.
* `topology_iterNNNN.vtu` — an **immutable per-iteration snapshot** of the alive
  mesh (sensitivity + von-Mises fields), so the topology evolution can be
  replayed/animated after the run. These are small (only the surviving tets).

Set **`beso.archive_iterations: true`** to also keep each iteration's key
OpenRadioss outputs under `work_dir/iter_NNNN/` before the `solve/` folder is
recycled for the next iteration: the mutated `<stem>_0000.rad`, the final
animation state(s) `<stem>A0*`, and the engine listing `<stem>_0001.out`.

> **Disk cost.** Archiving is off by default because it adds up: tens of MB per
> iteration (deck + animation), so a 50–150 iteration run can reach several GB.
> The ~345 MB restart (`<stem>_0000_0001.rst`) is deliberately **never** copied
> — that alone would be ~50 GB over a long run.

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
  BESO ranking/threshold, status/checkpoint round-trips, VTK extraction,
  per-iteration snapshot/archive file-writing, and the run-folder fallback +
  source-deck isolation.

This project consumes decks produced by the sibling `k_to_rad_converter`
(LS-DYNA → OpenRadioss); see that project for the conversion step.
```

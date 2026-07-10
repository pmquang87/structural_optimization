# oropt quickstart (~10 minutes)

A fast path from a fresh checkout to a running topology optimisation. oropt is
topology optimisation for **OpenRadioss TET4 solid parts**; the AlSi10Mg elevator
linkage is the reference example. For the full reference see [`README.md`](README.md).

> **Heads-up:** oropt drives the *real* OpenRadioss solver in the loop, so today
> you need a working OpenRadioss install (native Windows + Intel oneAPI MPI, **or**
> the Dockerised MUMPS build) and your own converted deck to actually run. A
> zero-solver `demo` backend is planned (roadmap P2/U2) but not here yet — until
> then this quickstart gets you installed, oriented, and launch-ready.

## 1. Install

A Python 3.10+ environment; the GUI extra pulls in vtk/scipy/streamlit:

```
python -m pip install -e .[gui]
```

Optional extras: `report3d` (interactive 3D viewer in `report.html`),
`growthmesh` (TetGen add-material meshing), `dev` (pytest/ruff).

Check it imports:

```
python -c "import oropt; print('ok')"
```

## 2. Look at the example config

Open [`configs/elevator_linkage.yaml`](configs/elevator_linkage.yaml) — it is
heavily commented and shows the current schema. The parts you will edit:

- `or_paths.root` — your native OpenRadioss install (or set `docker.enabled: true`
  to use the Docker MUMPS build and ignore `or_paths` entirely).
- `model.case_dir` — the folder holding your `<stem>_0000.rad` / `_0001.rad` decks.
- `model.design_part_id` / `design_node_min` / `bc_group_id` — which `/TETRA4`
  part is the design domain and which `/GRNOD/NODE` group to protect.
- `load_cases:` — each case's deck `stem`, `weight`, `sigma_allow` (stress limit)
  and `disp_constraints` (`{node_id, d_allow}`).

Validate a config (fast, ~1 s, no solver) before committing to a run:

```
python -c "from oropt.config import Config; import yaml; \
raw=yaml.safe_load(open('configs/elevator_linkage.yaml')); \
from oropt.validate import validate_config; \
[print(p) for p in validate_config(Config.from_dict(raw), raw=raw)]"
```

`error:` lines block the run; `warning:` lines let it proceed. On a fresh
checkout the only errors are "path/deck/executable not found" (the paths are
placeholders — point them at your deck and install).

Not sure which starting point to copy? See [`configs/presets/`](configs/presets/)
(coarse_proxy, final_highfidelity, multiload, additive, docker_mumps) and
[`docs/choosing_an_optimizer.md`](docs/choosing_an_optimizer.md).

## 3. Run it

Headless (CLI):

```
python -m oropt.run --config configs/elevator_linkage.yaml            # start
python -m oropt.run --config configs/elevator_linkage.yaml --resume   # resume
```

Or the GUI (configure, launch, live-monitor; safe to close mid-run):

```
streamlit run oropt/gui/app.py
# or: python run_gui.py   (PyCharm green-Run friendly)
```

The config check runs automatically at launch and aborts on any error (override
with `--skip-validate`).

## 4. What to expect as output

A run is ~13 min/solve × 50–150 iterations (11–33 h); per-iteration checkpoints
make it resumable, so develop on a coarse proxy mesh first (see
`configs/presets/coarse_proxy.yaml`). Written into the run folder (`work_dir`, or
`model.case_dir` when blank):

- `status.json` / `history.csv` — live state + one row per iteration.
- `topology_latest.vtu` and immutable `topology_iterNNNN.vtu` snapshots.
- `report.html` / `report.md` — auto summary (convergence charts, final design,
  σ_max/displacement vs limits, % mass removed).
- `topology_smoothed.stl` — the smoothed final surface (a CAD/print deliverable).
- `topology_evolution.gif` — material-removal animation.
- `d3plot/<stem>.d3plot` — LS-Dyna-viewable result per load case.
- `run.log` — the run's full log (also tailed in the GUI Monitor). Start here if
  anything looks wrong — see [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Next steps

- [`docs/choosing_an_optimizer.md`](docs/choosing_an_optimizer.md) — pick BESO /
  level-set / TOBS / HCA / SAIP for your problem.
- [`docs/applicability.md`](docs/applicability.md) — can I use this for my part?
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — common failure modes.
- [`README.md`](README.md) — the full reference (constraints, growth regions,
  manufacturing, back-off controller, outputs).

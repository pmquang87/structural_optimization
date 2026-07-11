# oropt quickstart (~10 minutes)

A fast path from a fresh checkout to a running topology optimisation. oropt is
topology optimisation for **OpenRadioss TET4 solid parts**; the AlSi10Mg elevator
linkage is the reference example. For the full reference see [`README.md`](README.md).

> **No OpenRadioss? Start anyway.** The zero-solver **demo backend** answers
> every solve with deterministic synthetic physics, so the whole pipeline —
> loop, GUI monitor, report, smoothing, GIF — runs on the bundled example in
> seconds (step 0 below). For a *real* optimisation you need a working
> OpenRadioss install (native Windows + Intel oneAPI MPI, **or** the Dockerised
> MUMPS build — see [`docs/docker_image.md`](docs/docker_image.md)) and your
> own converted deck.

## 0. Try it with zero solver (~2 minutes)

The repo bundles a ~2850-tet cantilever (`examples/cantilever/`) and a `demo:`
config section. After the install in step 1:

```
python - <<'EOF'
from oropt.config import Config
from oropt.loop import run_optimization
cfg = Config.from_dict({
    "demo": {"enabled": True},
    "model": {"case_dir": "examples/cantilever", "design_part_id": 60000000,
              "design_node_min": 60000000, "bc_group_id": 60000000},
    "load_cases": [{"name": "tip", "stem": "cantilever", "sigma_allow": 400.0,
                    "disp_constraints": [{"node_id": 60000699, "d_allow": 5.0}]}],
    "beso": {"target_volume_fraction": 0.6, "evolution_rate": 0.05,
             "filter_radius": 8.0, "max_iter": 15},
    "work_dir": "runs/demo",
})
print(run_optimization(cfg).state)
EOF
```

Then open `runs/demo/report.html` (and `topology_evolution.gif`). Demo numbers
are **synthetic** — it demonstrates and benchmarks the optimisers, it does not
analyse your part. `scripts/benchmark_optimizers.py` runs all five optimisers
head-to-head on this case.

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

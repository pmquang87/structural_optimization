# Benchmark & sweep harnesses (demo backend)

Two CLIs run the whole optimiser portfolio and its knobs against the bundled
`examples/cantilever` case using the **demo solver backend** (`oropt/demo.py`) —
deterministic synthetic physics, **zero OpenRadioss**, ~1 s per run:

- `scripts/benchmark_optimizers.py` — every optimiser (BESO / level-set / TOBS /
  HCA / SAIP) on the *same* problem with *identical* knobs, head to head (roadmap V4).
- `scripts/sweep.py` — one optimiser across a 1-D or 2-D grid of a knob
  (`evolution_rate`, `filter_radius`, `target_vf`, `max_iter`), with a robustness
  summary (roadmap V7).

> **The numbers are synthetic.** The demo backend concentrates energy along a
> pseudo load path and raises stress/displacement as material is removed; it does
> **not** analyse the part. So these harnesses benchmark optimiser *behaviour* —
> convergence discipline, oscillation, determinism, knob-sensitivity, the speed of
> the update machinery — **not** physics. A real-solver benchmark against a
> literature-known optimum (a cantilever/MBB/L-bracket with a published compliance
> curve, roadmap V3) is still future work and would run through the same harness by
> flipping the backend off `demo`.

## Running

```bash
# all five optimisers, per-run wall budget 30 s, results under bench/
python scripts/benchmark_optimizers.py --max-iter 12 --out bench

# a subset with explicit shared knobs
python scripts/benchmark_optimizers.py --optimizers beso,hca,saip \
    --target-vf 0.6 --evolution-rate 0.05 --filter-radius 8 --out bench

# sweep evolution_rate (1-D) or evolution_rate x filter_radius (2-D)
python scripts/sweep.py --param evolution_rate --values 0.02,0.05,0.1 --out sweep
python scripts/sweep.py --optimizer hca --param filter_radius --values 4,8,12 \
    --param2 evolution_rate --values2 0.02,0.05 --out sweep2d
```

Each optimiser/cell runs in an **isolated child process bounded by `--timeout`**
(default 120 s): a run that exceeds it is recorded `state: timeout` and the batch
continues, so one slow optimiser can never hang the harness. Outputs: a markdown
table (also printed), a machine-readable CSV, and — for the benchmark —
`convergence.csv` (long-format per-iteration traces for overlay plots).

## First head-to-head results (demo mode)

`benchmark_optimizers.py --max-iter 12` (target_vf 0.7, evolution_rate 0.05,
filter_radius 8, per-run timeout 30 s):

| optimizer | state | iters | final vf | sigma_max | disp | feasible | mask hash |
|---|---|---|---|---|---|---|---|
| beso | converged | 11 | 0.7000 | 170.75 | 0.854 | True | ffd9efc6df45 |
| levelset | converged | 11 | 0.3067 | 588.84 | 2.944 | False | 47b40a4e537d |
| tobs | timeout | – | – | – | – | – | – |
| hca | converged | 11 | 0.7000 | 170.75 | 0.854 | True | 2d10010e0537 |
| saip | converged | 11 | 0.7000 | 170.75 | 0.854 | True | ffd9efc6df45 |

What this demo-mode run actually shows (behaviour, not physics):

- **BESO, HCA and SAIP agree.** All three hit the `target_vf = 0.7` gate and report
  the identical peak/displacement; **BESO and SAIP converge to the byte-identical
  design** (same mask hash), and HCA lands on the same volume/stress by a different
  internal path. Three independent update rules agreeing on the same problem is the
  cross-check the portfolio never had before.
- **Level-set overshoots.** Its φ-evolution drove well past the volume gate to
  vf ≈ 0.31 and ended **infeasible** — on this monotone synthetic response the
  boundary-evolution rule is far more aggressive than the discrete carvers. A real
  finding to watch (align `dt`/`target_volume_fraction` when staging into level-set;
  see `docs/optimizer_switching.md`).
- **TOBS does not scale to this mesh.** Its per-iteration integer program
  (`scipy.optimize.milp` / HiGHS over ~2850 binary flip variables with a move-limit
  and an ε-relaxed volume band) does not return within the 30 s budget — the demo
  cantilever is already too fine for the ILP. TOBS is best kept for coarse proxy
  meshes; this is exactly the kind of scaling limit the head-to-head was meant to
  surface.

## A robustness finding from the sweep

`sweep.py --param filter_radius --values 4,8,12 --optimizer beso`:

| filter_radius | state | iters | final vf | sigma_max | feasible |
|---|---|---|---|---|---|
| 4.0 | converged | 11 | 0.7000 | 170.75 | True |
| 8.0 | converged | 11 | 0.7000 | 170.75 | True |
| 12.0 | converged | 19 | 0.1888 | 1219.25 | False |

BESO is stable across `evolution_rate` (same design, fewer iterations at higher
rates) but **a too-large `filter_radius` destabilises it**: at radius 12 on this
mesh the smeared sensitivity drove the design past the gate to vf ≈ 0.19 and
**infeasible**. The README's "BESO is heuristic — sensitive to filter radius"
caveat, now measured. Keep the filter radius a small multiple of the element size.

## Notes

- `scripts/` is not a package; `sweep.py` imports the shared demo-config recipe and
  the timeout-isolated runner from `benchmark_optimizers.py` via a `sys.path` insert.
- Determinism: every optimiser and the demo backend are deterministic, so re-running
  reproduces the same `mask hash` — the hash column is a regression fingerprint.
- Coverage: `tests/test_benchmark.py` exercises both CLIs (including the timeout path)
  hermetically.

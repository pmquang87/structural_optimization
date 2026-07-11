"""Cross-optimizer benchmark harness on the demo solver backend (roadmap V4).

Runs the SAME bundled ``examples/cantilever`` case through each selected
optimizer (beso / levelset / tobs / hca / saip) with identical knobs, using the
synthetic ``demo`` backend (zero OpenRadioss, ~1-2 s per run), and tabulates
per-run: final state, iterations, final volume fraction, sigma_max, disp,
feasibility, wall seconds and a determinism hash of the final alive mask.

The numbers are SYNTHETIC (see ``oropt/demo.py``): this benchmarks optimizer
BEHAVIOUR -- convergence discipline, oscillation, determinism, speed of the
update machinery -- not physics. The real-solver benchmark on a
literature-known optimum (roadmap V3) remains future work.

Outputs (under ``--out``):
  * ``benchmark.md``    -- the markdown table (also printed to stdout)
  * ``benchmark.csv``   -- the same rows, machine-readable
  * ``convergence.csv`` -- long-format per-iteration overlay data
                           (optimizer, iteration, volume_fraction, sigma_max, disp)
  * ``runs/<name>/``    -- each optimizer's private case copy + run dir

Example:
    python scripts/benchmark_optimizers.py --max-iter 5 --out /tmp/bench
    python scripts/benchmark_optimizers.py --optimizers beso,tobs --target-vf 0.6

Exit code is 0 if at least one optimizer completed without failing.

NOTE: ``scripts/sweep.py`` (roadmap V7) imports this module for the shared
``build_demo_config`` / ``run_one`` helpers via a ``sys.path`` insertion --
``scripts/`` is not a package, so that is the cleanest way to share the code
without duplicating it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:      # allow running without `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT))

from oropt.config import Config                     # noqa: E402
from oropt.loop import run_optimization             # noqa: E402
from oropt.status import load_checkpoint, read_history  # noqa: E402

EXAMPLE = REPO_ROOT / "examples" / "cantilever"
OPTIMIZERS = ("beso", "levelset", "tobs", "hca", "saip")

#: benchmark.csv / row-dict columns, in output order.
ROW_COLS = ["optimizer", "state", "iterations", "volume_fraction", "sigma_max",
            "disp", "feasible", "elements_alive", "wall_s", "mask_sha256",
            "message"]
CONVERGENCE_COLS = ["optimizer", "iteration", "volume_fraction", "sigma_max", "disp"]


def build_demo_config(root: Path, optimizer: str, target_vf: float,
                      evolution_rate: float, filter_radius: float,
                      max_iter: int, overrides: dict | None = None) -> Config:
    """The exact demo-backend recipe proven by ``tests/test_demo_wiring.py``:
    a private copy of the bundled cantilever under *root*/case, ``demo.enabled``,
    one tip load case, post-processing off. The shared knobs (and any extra
    *overrides*, e.g. a sweep's parameter) are applied identically to EVERY
    optimizer block -- the blocks mirror each other, and only the active one is
    read -- so switching ``optimizer`` changes nothing else.
    """
    case_dir = root / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(EXAMPLE.glob("cantilever_000*.rad")):
        shutil.copy2(f, case_dir / f.name)
    block = {"target_volume_fraction": target_vf, "evolution_rate": evolution_rate,
             "filter_radius": filter_radius, "max_iter": max_iter,
             "protect_layers": 1, "archive_iterations": False}
    block.update(overrides or {})
    return Config.from_dict({
        "demo": {"enabled": True},
        "model": {"case_dir": str(case_dir), "design_part_id": 60000000,
                  "design_node_min": 60000000, "bc_group_id": 60000000},
        "load_cases": [{"name": "tip", "stem": "cantilever",
                        "sigma_allow": 400.0,
                        "disp_constraints": [{"node_id": 60000699,
                                              "d_allow": 5.0}]}],
        "optimizer": optimizer,
        **{name: dict(block) for name in OPTIMIZERS},
        "d3plot": {"enabled": False}, "smooth": {"enabled": False},
        "animate": {"enabled": False}, "report": {"enabled": False},
        "work_dir": str(root / "run"),
    })


def final_mask_hash(work_dir: str | Path) -> str:
    """sha256 of the checkpointed final alive mask (determinism fingerprint).

    Two runs with identical inputs must produce identical hashes -- the demo
    backend and every optimizer update are deterministic by construction.
    Empty string when no checkpoint was written (e.g. failure before iter 1).
    """
    try:
        ck = load_checkpoint(work_dir)
    except Exception:
        return ""
    if ck is None:
        return ""
    mask = np.asarray(ck["alive_mask"], dtype=bool)
    return hashlib.sha256(mask.tobytes()).hexdigest()


def run_one(cfg: Config, verbose: bool = False) -> dict:
    """Run one optimization and return its benchmark row (never raises)."""
    log = print if verbose else (lambda _msg: None)
    name = cfg.optimizer_name()
    t0 = time.perf_counter()
    try:
        status = run_optimization(cfg, log=log)
        row = {"optimizer": name, "state": status.state,
               "iterations": status.iteration,
               "volume_fraction": status.volume_fraction,
               "sigma_max": status.sigma_max, "disp": status.disp,
               "feasible": status.feasible,
               "elements_alive": status.elements_alive,
               "message": status.message}
    except Exception as exc:   # one optimizer failing must not sink the batch
        row = {"optimizer": name, "state": "failed", "iterations": 0,
               "volume_fraction": float("nan"), "sigma_max": float("nan"),
               "disp": float("nan"), "feasible": False, "elements_alive": 0,
               "message": f"{type(exc).__name__}: {exc}"}
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    row["mask_sha256"] = final_mask_hash(cfg.work_dir)
    return row


def run_isolated(name: str, out_root: Path, target_vf: float,
                 evolution_rate: float, filter_radius: float, max_iter: int,
                 timeout_s: float) -> tuple[dict, list[dict]]:
    """Run one optimizer in a child process bounded by *timeout_s*.

    Isolation keeps one optimizer's cost — notably TOBS, whose per-iteration
    integer program does not scale to fine meshes and can run for minutes — from
    hanging the whole batch: on timeout the child is killed and a ``state:
    "timeout"`` row is recorded (a real benchmark result, not a crash). The child
    prints ``{"row":..., "conv":...}`` as JSON; the parent parses its stdout.
    """
    work = out_root / "runs" / name
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_single", name,
           "--out", str(work), "--target-vf", str(target_vf),
           "--evolution-rate", str(evolution_rate),
           "--filter-radius", str(filter_radius), "--max-iter", str(max_iter)]
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return ({"optimizer": name, "state": "timeout", "iterations": 0,
                 "volume_fraction": float("nan"), "sigma_max": float("nan"),
                 "disp": float("nan"), "feasible": False, "elements_alive": 0,
                 "wall_s": round(time.perf_counter() - t0, 2), "mask_sha256": "",
                 "message": f"exceeded --timeout {timeout_s:g}s "
                            "(per-iteration solve did not scale)"}, [])
    if cp.returncode != 0:
        return ({"optimizer": name, "state": "failed", "iterations": 0,
                 "volume_fraction": float("nan"), "sigma_max": float("nan"),
                 "disp": float("nan"), "feasible": False, "elements_alive": 0,
                 "wall_s": round(time.perf_counter() - t0, 2), "mask_sha256": "",
                 "message": f"child exit {cp.returncode}: "
                            f"{cp.stderr.strip()[-200:]}"}, [])
    try:
        payload = json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return ({"optimizer": name, "state": "failed", "iterations": 0,
                 "volume_fraction": float("nan"), "sigma_max": float("nan"),
                 "disp": float("nan"), "feasible": False, "elements_alive": 0,
                 "wall_s": round(time.perf_counter() - t0, 2), "mask_sha256": "",
                 "message": "could not parse child output"}, [])
    return payload["row"], payload["conv"]


def history_rows(optimizer: str, work_dir: str | Path) -> list[dict]:
    """This run's history.csv reduced to the long-format convergence columns."""
    out = []
    for h in read_history(work_dir):
        out.append({"optimizer": optimizer,
                    "iteration": h.get("iteration", ""),
                    "volume_fraction": h.get("volume_fraction", ""),
                    "sigma_max": h.get("sigma_max", ""),
                    "disp": h.get("disp", "")})
    return out


def _fmt(v, nd: int) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def markdown_table(rows: list[dict]) -> str:
    """The benchmark rows as a compact GitHub-flavoured markdown table."""
    head = ["optimizer", "state", "iters", "final vf", "sigma_max", "disp",
            "feasible", "wall [s]", "mask hash"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["optimizer"], r["state"], str(r["iterations"]),
            _fmt(r["volume_fraction"], 4), _fmt(r["sigma_max"], 2),
            _fmt(r["disp"], 3), str(r["feasible"]), _fmt(r["wall_s"], 2),
            (r["mask_sha256"][:12] or "-"),
        ]) + " |")
    return "\n".join(lines)


def write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--optimizers", default=",".join(OPTIMIZERS),
                    help=f"comma-separated subset of {','.join(OPTIMIZERS)} "
                         "(default: all five)")
    ap.add_argument("--out", default="benchmark_out",
                    help="output directory (default: ./benchmark_out)")
    ap.add_argument("--max-iter", type=int, default=20,
                    help="max_iter for every optimizer (default 20)")
    ap.add_argument("--target-vf", type=float, default=0.7,
                    help="target_volume_fraction for every optimizer (default 0.7)")
    ap.add_argument("--evolution-rate", type=float, default=0.05,
                    help="evolution_rate for every optimizer (default 0.05)")
    ap.add_argument("--filter-radius", type=float, default=8.0,
                    help="filter_radius [mm] for every optimizer (default 8.0)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-optimizer wall-clock budget [s] (default 120); a "
                         "run exceeding it is recorded state=timeout and the batch "
                         "continues (TOBS's integer program can be slow on fine meshes)")
    ap.add_argument("--verbose", action="store_true",
                    help="stream each run's loop log instead of one summary line")
    # hidden: run ONE optimizer in-process and print its {row, conv} as JSON --
    # the mechanism run_isolated() uses to bound each optimizer in a child.
    ap.add_argument("--_single", metavar="NAME", default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args._single is not None:
        cfg = build_demo_config(Path(args.out), args._single, args.target_vf,
                                args.evolution_rate, args.filter_radius,
                                args.max_iter)
        row = run_one(cfg, verbose=False)
        conv = history_rows(args._single, cfg.work_dir)
        print(json.dumps({"row": row, "conv": conv}))
        return 0

    names = [s.strip().lower() for s in args.optimizers.split(",") if s.strip()]
    bad = [n for n in names if n not in OPTIMIZERS]
    if bad:
        ap.error(f"unknown optimizer(s) {bad}; choose from {list(OPTIMIZERS)}")
    if not (EXAMPLE / "cantilever_0000.rad").is_file():
        print(f"error: bundled cantilever example missing under {EXAMPLE}",
              file=sys.stderr)
        return 1

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, conv = [], []
    for name in names:
        row, crows = run_isolated(name, out, args.target_vf, args.evolution_rate,
                                  args.filter_radius, args.max_iter, args.timeout)
        rows.append(row)
        conv.extend(crows)
        print(f"[bench] {name:<8} state={row['state']:<9} "
              f"iters={row['iterations']:<3} vf={_fmt(row['volume_fraction'], 4)} "
              f"sigma={_fmt(row['sigma_max'], 2)} wall={row['wall_s']}s")

    table = markdown_table(rows)
    print("\n" + table)
    (out / "benchmark.md").write_text(
        "# Cross-optimizer benchmark (demo backend -- SYNTHETIC physics)\n\n"
        f"Case: examples/cantilever | target_vf={args.target_vf} "
        f"evolution_rate={args.evolution_rate} filter_radius={args.filter_radius} "
        f"max_iter={args.max_iter}\n\n" + table + "\n",
        encoding="utf-8")
    write_csv(out / "benchmark.csv", ROW_COLS, rows)
    write_csv(out / "convergence.csv", CONVERGENCE_COLS, conv)
    print(f"\n[bench] wrote {out / 'benchmark.md'}, benchmark.csv, convergence.csv")
    return 0 if any(r["state"] != "failed" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

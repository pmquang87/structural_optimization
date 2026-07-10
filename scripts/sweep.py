"""Parameter-sweep harness on the demo solver backend (roadmap V7).

Grid-runs the bundled ``examples/cantilever`` demo case across a 1-D or 2-D grid
of one optimizer's knobs and tabulates how the outcome (final volume fraction,
peak stress, feasibility, iterations) moves with the parameter — the sensitivity
study the README calls for but never shipped ("BESO is heuristic — sensitive to
evolution rate, filter radius and history weight").

Numbers are SYNTHETIC (see ``oropt/demo.py``): this measures the optimiser's
*response to its own knobs* (robustness, monotonicity, failure cells), not
physics. Reuses ``benchmark_optimizers`` for the demo-config recipe and the
timeout-isolated per-cell run, so a pathological cell (e.g. TOBS's integer
program on a fine mesh) records ``state: timeout`` instead of hanging the sweep.

The swept knob(s) must be among ``evolution_rate``, ``filter_radius``,
``target_vf``, ``max_iter`` (the dimensions the isolated runner already carries).

Examples:
    python scripts/sweep.py --param evolution_rate --values 0.02,0.05,0.1
    python scripts/sweep.py --optimizer hca --param filter_radius --values 4,8,12 \
        --param2 evolution_rate --values2 0.02,0.05 --out /tmp/sweep
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# scripts/ is not a package; add it so the shared harness imports cleanly.
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_optimizers import OPTIMIZERS, _fmt, run_isolated, write_csv  # noqa: E402

#: knobs the isolated runner carries, and their run_isolated keyword.
SWEEPABLE = {"evolution_rate": "evolution_rate", "filter_radius": "filter_radius",
             "target_vf": "target_vf", "max_iter": "max_iter"}
DEFAULTS = {"target_vf": 0.7, "evolution_rate": 0.05, "filter_radius": 8.0,
            "max_iter": 20}


def _parse_values(param: str, raw: str) -> list:
    cast = int if param == "max_iter" else float
    return [cast(v.strip()) for v in raw.split(",") if v.strip()]


def run_cell(name: str, out: Path, knobs: dict, timeout_s: float) -> dict:
    """One grid cell: run_isolated with the swept knobs overlaid on the defaults."""
    k = dict(DEFAULTS)
    k.update(knobs)
    row, _conv = run_isolated(name, out, k["target_vf"], k["evolution_rate"],
                              k["filter_radius"], int(k["max_iter"]), timeout_s)
    row.update({p: knobs[p] for p in knobs})     # tag the row with its cell coords
    return row


def robustness(rows: list[dict], swept: list[str]) -> list[str]:
    """Human-readable robustness notes: spread of the outcome across the grid and
    any failed/timeout/infeasible cells (the signals a sweep is meant to expose)."""
    ok = [r for r in rows if r["state"] not in ("failed", "timeout")]
    notes = []
    bad = [r for r in rows if r["state"] in ("failed", "timeout")]
    if bad:
        notes.append(f"{len(bad)}/{len(rows)} cell(s) did not complete: "
                     + ", ".join(f"{{{', '.join(f'{p}={r[p]}' for p in swept)}}}"
                                 f"={r['state']}" for r in bad))
    infeas = [r for r in ok if not r["feasible"]]
    if infeas:
        notes.append(f"{len(infeas)}/{len(ok)} completed cell(s) ended INFEASIBLE")
    vfs = [r["volume_fraction"] for r in ok
           if isinstance(r["volume_fraction"], (int, float))]
    if len(vfs) >= 2:
        notes.append(f"final vf across the grid: min={min(vfs):.4f} "
                     f"max={max(vfs):.4f} stdev={statistics.pstdev(vfs):.4f}")
    sigs = [r["sigma_max"] for r in ok
            if isinstance(r["sigma_max"], (int, float))]
    if len(sigs) >= 2:
        notes.append(f"final sigma_max: min={min(sigs):.2f} max={max(sigs):.2f} "
                     f"stdev={statistics.pstdev(sigs):.2f}")
    return notes


def markdown(rows: list[dict], swept: list[str]) -> str:
    head = swept + ["state", "iters", "final vf", "sigma_max", "feasible", "wall [s]"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        cells = [str(r[p]) for p in swept] + [
            r["state"], str(r["iterations"]), _fmt(r["volume_fraction"], 4),
            _fmt(r["sigma_max"], 2), str(r["feasible"]), _fmt(r["wall_s"], 2)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--optimizer", default="beso",
                    help=f"one of {','.join(OPTIMIZERS)} (default beso; note TOBS "
                         "may time out on the demo mesh)")
    ap.add_argument("--param", required=True, choices=sorted(SWEEPABLE),
                    help="knob to sweep")
    ap.add_argument("--values", required=True,
                    help="comma-separated values for --param")
    ap.add_argument("--param2", choices=sorted(SWEEPABLE), default=None,
                    help="optional second knob for a 2-D grid")
    ap.add_argument("--values2", default=None, help="values for --param2")
    ap.add_argument("--out", default="sweep_out", help="output directory")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-cell wall-clock budget [s] (default 120)")
    args = ap.parse_args(argv)

    if args.optimizer not in OPTIMIZERS:
        ap.error(f"unknown optimizer {args.optimizer!r}; choose from {list(OPTIMIZERS)}")
    if args.param2 and not args.values2:
        ap.error("--param2 requires --values2")
    if args.param2 == args.param:
        ap.error("--param2 must differ from --param")

    grid1 = _parse_values(args.param, args.values)
    grid2 = _parse_values(args.param2, args.values2) if args.param2 else [None]
    swept = [args.param] + ([args.param2] if args.param2 else [])

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, v1 in enumerate(grid1):
        for j, v2 in enumerate(grid2):
            knobs = {args.param: v1}
            if args.param2:
                knobs[args.param2] = v2
            row = run_cell(args.optimizer, out / "cells" / f"c{i}_{j}", knobs,
                           args.timeout)
            rows.append(row)
            coords = " ".join(f"{p}={knobs[p]}" for p in swept)
            print(f"[sweep] {coords:<32} state={row['state']:<9} "
                  f"vf={_fmt(row['volume_fraction'], 4)} "
                  f"sigma={_fmt(row['sigma_max'], 2)} wall={row['wall_s']}s")

    table = markdown(rows, swept)
    notes = robustness(rows, swept)
    print("\n" + table)
    if notes:
        print("\nRobustness:")
        for n in notes:
            print(f"  - {n}")

    cols = swept + ["state", "iterations", "volume_fraction", "sigma_max", "disp",
                    "feasible", "elements_alive", "wall_s", "mask_sha256", "message"]
    write_csv(out / "sweep.csv", cols, rows)
    (out / "sweep.md").write_text(
        f"# Parameter sweep ({args.optimizer}, demo backend -- SYNTHETIC)\n\n"
        + table + "\n\n## Robustness\n\n"
        + ("\n".join(f"- {n}" for n in notes) if notes else "- (single cell)")
        + "\n", encoding="utf-8")
    print(f"\n[sweep] wrote {out / 'sweep.md'}, sweep.csv")
    return 0 if any(r["state"] not in ("failed", "timeout") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

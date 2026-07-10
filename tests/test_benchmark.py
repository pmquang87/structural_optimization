"""Smoke tests for the demo-backend benchmark & sweep harnesses (roadmap V4/V7).

Exercise the real CLIs via subprocess so the argument parsing, the isolated
per-run child mode, and the CSV/markdown emission are all covered. Kept fast:
demo runs are ~1 s and the fast optimizers (beso/hca) are used with a short
per-run timeout. TOBS is deliberately excluded — it times out on the demo mesh
(a real finding documented in docs/benchmarks.md), which would only slow the
test without adding coverage the timeout path (tested separately) doesn't."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "scripts" / "benchmark_optimizers.py"
SWEEP = REPO / "scripts" / "sweep.py"


@pytest.fixture(scope="module")
def example_present():
    if not (REPO / "examples" / "cantilever" / "cantilever_0000.rad").is_file():
        pytest.skip("bundled cantilever example missing")


def _run(cmd, timeout=180):
    return subprocess.run([sys.executable, *cmd], capture_output=True, text=True,
                          timeout=timeout, cwd=str(REPO))


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_benchmark_cli(tmp_path, example_present):
    out = tmp_path / "bench"
    cp = _run([str(BENCH), "--optimizers", "beso,hca", "--max-iter", "3",
               "--timeout", "60", "--out", str(out)])
    assert cp.returncode == 0, cp.stderr
    assert (out / "benchmark.md").is_file()
    rows = _read_csv(out / "benchmark.csv")
    names = {r["optimizer"] for r in rows}
    assert names == {"beso", "hca"}
    # at least one optimizer completed (not failed/timeout)
    assert any(r["state"] not in ("failed", "timeout") for r in rows)
    # convergence long-format data was written with per-iteration rows
    conv = _read_csv(out / "convergence.csv")
    assert conv and {"optimizer", "iteration", "volume_fraction"} <= set(conv[0])


def test_benchmark_timeout_is_recorded(tmp_path, example_present):
    """A pathological optimizer (TOBS's MILP on the demo mesh) hits the per-run
    timeout and is recorded state=timeout instead of hanging the batch."""
    out = tmp_path / "bench_to"
    _run([str(BENCH), "--optimizers", "tobs", "--max-iter", "5",
          "--timeout", "3", "--out", str(out)], timeout=60)
    # exit 1 (no optimizer succeeded) is fine; the point is it returns promptly
    rows = _read_csv(out / "benchmark.csv")
    assert len(rows) == 1 and rows[0]["optimizer"] == "tobs"
    assert rows[0]["state"] == "timeout"


def test_sweep_cli_1d(tmp_path, example_present):
    out = tmp_path / "sweep"
    cp = _run([str(SWEEP), "--param", "evolution_rate", "--values", "0.05,0.1",
               "--optimizer", "beso", "--out", str(out), "--timeout", "60"])
    assert cp.returncode == 0, cp.stderr
    assert (out / "sweep.md").is_file()
    rows = _read_csv(out / "sweep.csv")
    assert len(rows) == 2                          # 2 grid cells
    assert {r["evolution_rate"] for r in rows} == {"0.05", "0.1"}
    assert all(r["state"] == "converged" for r in rows)


def test_sweep_cli_2d(tmp_path, example_present):
    out = tmp_path / "sweep2"
    cp = _run([str(SWEEP), "--param", "evolution_rate", "--values", "0.05,0.1",
               "--param2", "filter_radius", "--values2", "6,10",
               "--optimizer", "beso", "--out", str(out), "--timeout", "60"])
    assert cp.returncode == 0, cp.stderr
    rows = _read_csv(out / "sweep.csv")
    assert len(rows) == 4                          # 2x2 grid

"""Run-history browser / multi-run compare helpers (Streamlit-free, hermetic).

Fabricates run folders in tmp_path with the *real* on-disk formats — status.json
via json over :class:`oropt.status.Status` field names, history.csv via
:func:`oropt.status.append_history` (the loop's own writer), config_used.yaml as
plain YAML — and asserts the pure helpers in :mod:`oropt.gui.history` exactly.
No Streamlit / AppTest anywhere, so this runs on every CI leg.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from oropt import status as st_io
from oropt.gui.history import (ARTIFACTS, SERIES_KEYS, RunEntry, artifact_mime,
                               compare_table, downloadable_artifacts,
                               entry_label, load_history_series, overlay_series,
                               scan_runs)


# ---- fixtures ---------------------------------------------------------------
def make_run(root: Path, rel: str, *, state: str = "converged",
             iteration: int = 30, max_iter: int = 60, vf: float = 0.35,
             sigma: float = 245.2, feasible: bool = True,
             updated: str = "2026-07-01T10:00:00",
             optimizer: str | None = None) -> Path:
    """A minimal but format-true run folder: status.json (+config_used.yaml)."""
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({
        "state": state, "iteration": iteration, "max_iter": max_iter,
        "volume_fraction": vf, "sigma_max": sigma, "feasible": feasible,
        "updated": updated, "message": "",
    }), encoding="utf-8")
    if optimizer is not None:
        (d / "config_used.yaml").write_text(
            f"optimizer: {optimizer}\nwork_dir: runs/x\n", encoding="utf-8")
    return d


def history_row(it: int, vf: float, sigma: float, disp: float,
                feasible: bool) -> dict:
    return {"iteration": it, "volume_fraction": vf, "sigma_max": sigma,
            "disp": disp, "elements_alive": 100, "feasible": feasible,
            "iter_wall_s": 1.0, "or_termination": "NORMAL", "optimizer": "beso"}


# ---- scan_runs --------------------------------------------------------------
def test_scan_finds_runs_and_reads_headlines(tmp_path):
    make_run(tmp_path, "run_a", state="converged", iteration=60, max_iter=60,
             vf=0.30, sigma=240.0, feasible=True,
             updated="2026-07-01T10:00:00", optimizer="beso")
    make_run(tmp_path, "run_b", state="running", iteration=5, max_iter=40,
             vf=0.88, sigma=190.5, feasible=True,
             updated="2026-07-02T09:00:00")                 # no config_used.yaml
    make_run(tmp_path, "run_c", state="failed", iteration=3, max_iter=40,
             vf=0.95, sigma=512.0, feasible=False,
             updated="2026-06-30T08:00:00", optimizer="levelset")

    entries = scan_runs(tmp_path)
    assert [e.name for e in entries] == ["run_b", "run_a", "run_c"]  # newest first
    b, a, c = entries
    assert a == RunEntry(path=tmp_path / "run_a", name="run_a",
                         state="converged", iteration=60, max_iter=60,
                         volume_fraction=0.30, sigma_max=240.0, feasible=True,
                         optimizer="beso", updated="2026-07-01T10:00:00")
    assert b.state == "running" and b.optimizer == ""    # tolerated absence
    assert c.state == "failed" and not c.feasible and c.optimizer == "levelset"


def test_scan_skips_malformed_runs(tmp_path):
    make_run(tmp_path, "good", updated="2026-07-01T10:00:00")
    bad = tmp_path / "bad_json"
    bad.mkdir()
    (bad / "status.json").write_text("{not json", encoding="utf-8")
    unknown = tmp_path / "bad_key"                        # unknown field -> skipped
    unknown.mkdir()
    (unknown / "status.json").write_text(json.dumps({"state": "x", "bogus": 1}),
                                         encoding="utf-8")
    assert [e.name for e in scan_runs(tmp_path)] == ["good"]


def test_scan_depth_is_bounded(tmp_path):
    make_run(tmp_path, "a/b/c", updated="2026-07-01T10:00:00")       # depth 3: found
    make_run(tmp_path, "a/b/c/d", updated="2026-07-02T10:00:00")     # depth 4: not
    assert [e.name for e in scan_runs(tmp_path)] == ["a/b/c"]


def test_scan_root_itself_can_be_a_run(tmp_path):
    make_run(tmp_path, ".", state="stopped")
    [e] = scan_runs(tmp_path)
    assert e.name == "." and e.path == tmp_path and e.state == "stopped"


def test_scan_respects_limit_keeping_newest(tmp_path):
    for i, day in enumerate(("01", "03", "02")):
        make_run(tmp_path, f"run_{i}", updated=f"2026-07-{day}T10:00:00")
    entries = scan_runs(tmp_path, limit=2)
    assert [e.name for e in entries] == ["run_1", "run_2"]


def test_scan_missing_root_reads_as_no_runs(tmp_path):
    assert scan_runs(tmp_path / "nope") == []
    assert scan_runs(tmp_path) == []                       # empty dir, no runs


def test_blank_updated_falls_back_to_file_mtime(tmp_path):
    make_run(tmp_path, "old", updated="")
    [e] = scan_runs(tmp_path)
    assert e.updated                                       # non-empty ISO text
    assert e.updated[:4].isdigit() and "T" in e.updated


# ---- load_history_series ----------------------------------------------------
def test_history_series_from_real_writer(tmp_path):
    d = make_run(tmp_path, "run")
    st_io.append_history(d, history_row(1, 0.95, 210.0, 1.10, True))
    st_io.append_history(d, history_row(2, 0.90, 250.5, 1.25, False))
    assert load_history_series(d) == {
        "iteration": [1, 2],
        "volume_fraction": [0.95, 0.90],
        "sigma_max": [210.0, 250.5],
        "disp": [1.10, 1.25],
        "feasible": [True, False],
    }


def test_history_series_missing_file_is_empty(tmp_path):
    assert load_history_series(tmp_path) == {k: [] for k in SERIES_KEYS}


def test_history_series_tolerates_bad_cells(tmp_path):
    d = tmp_path
    (d / "history.csv").write_text(
        "iteration,volume_fraction,sigma_max,disp,feasible\n"
        "1,0.9,oops,1.0,True\n"          # bad float cell -> NaN, row kept aligned
        "oops,0.8,200,1.1,True\n"        # bad iteration -> whole row dropped
        "2,0.8,205.0,1.1,False\n", encoding="utf-8")
    s = load_history_series(d)
    assert s["iteration"] == [1, 2]
    assert s["volume_fraction"] == [0.9, 0.8]
    assert math.isnan(s["sigma_max"][0]) and s["sigma_max"][1] == 205.0
    assert s["feasible"] == [True, False]


def test_overlay_series_shapes_runs_for_chart():
    named = {
        "run_a": {"iteration": [1, 2], "sigma_max": [210.0, 250.5]},
        "run_b": {"iteration": [1, 2, 3], "sigma_max": [float("nan"), 300.0, 280.0]},
        "run_c": {"iteration": [], "sigma_max": []},       # no data -> dropped
    }
    assert overlay_series(named, "sigma_max") == {
        "run_a": {1: 210.0, 2: 250.5},
        "run_b": {2: 300.0, 3: 280.0},                     # NaN sample dropped
    }


# ---- compare_table ----------------------------------------------------------
def test_compare_table_rows(tmp_path):
    make_run(tmp_path, "run_a", state="converged", iteration=60, max_iter=60,
             vf=0.30, sigma=240.0, feasible=True,
             updated="2026-07-02T10:00:00", optimizer="beso")
    make_run(tmp_path, "run_b", state="failed", iteration=3, max_iter=40,
             vf=0.95, sigma=512.0, feasible=False, updated="2026-07-01T10:00:00")
    rows = compare_table(scan_runs(tmp_path))
    assert rows == [
        {"run": "run_a", "state": "✅ converged", "optimizer": "beso",
         "iteration": "60/60", "volume_fraction": 0.30, "σ_max [MPa]": 240.0,
         "feasible": "✅", "updated": "2026-07-02T10:00:00"},
        {"run": "run_b", "state": "❌ failed", "optimizer": "—",
         "iteration": "3/40", "volume_fraction": 0.95, "σ_max [MPa]": 512.0,
         "feasible": "⚠️", "updated": "2026-07-01T10:00:00"},
    ]


def test_entry_label_headline(tmp_path):
    make_run(tmp_path, "run_a", state="running", iteration=5, max_iter=40,
             vf=0.875, sigma=190.54, feasible=True, optimizer="tobs")
    [e] = scan_runs(tmp_path)
    assert entry_label(e) == ("🟢 run_a · it 5/40 · vf 0.875 · σ 190.5 MPa "
                              "· ✅ · tobs")


def test_entry_label_nan_sigma_shows_dash(tmp_path):
    make_run(tmp_path, "fresh", state="idle", iteration=0, max_iter=40,
             vf=1.0, sigma=float("nan"), feasible=False)
    [e] = scan_runs(tmp_path)
    assert "σ — MPa" in entry_label(e) and "⚠️" in entry_label(e)


# ---- downloadable_artifacts -------------------------------------------------
def test_downloadable_artifacts_lists_only_existing(tmp_path):
    for fname in ("report.html", "topology_smoothed.stl",
                  "topology_evolution.gif", "history.csv",
                  "manufacturability.json"):                 # no .vtp on purpose
        (tmp_path / fname).write_text("x", encoding="utf-8")
    (tmp_path / "topology_smoothed.vtp").mkdir()             # a DIR must not list
    arts = downloadable_artifacts(tmp_path)
    assert [(label, p.name) for label, p in arts] == [
        ("📝 Report", "report.html"),
        ("🧊 Smoothed surface (STL)", "topology_smoothed.stl"),
        ("🎬 Evolution GIF", "topology_evolution.gif"),
        ("📈 Iteration history (CSV)", "history.csv"),
        ("🏭 Manufacturability audit", "manufacturability.json"),
    ]
    assert all(p.parent == tmp_path for _l, p in arts)


def test_downloadable_artifacts_empty_dir(tmp_path):
    assert downloadable_artifacts(tmp_path) == []


def test_artifact_catalog_matches_writer_constants():
    # the catalogue must track the writers' own filename constants
    from oropt.animate import ANIM_GIF
    from oropt.report import REPORT_HTML
    from oropt.smoothing import SMOOTHED_BASE
    names = [fname for _label, fname in ARTIFACTS]
    assert REPORT_HTML in names and ANIM_GIF in names and st_io.HISTORY in names
    assert f"{SMOOTHED_BASE}.stl" in names and f"{SMOOTHED_BASE}.vtp" in names


def test_artifact_mime_types():
    assert artifact_mime("report.html") == "text/html"
    assert artifact_mime(Path("x/topology_evolution.gif")) == "image/gif"
    assert artifact_mime("history.csv") == "text/csv"
    assert artifact_mime("manufacturability.json") == "application/json"
    assert artifact_mime("topology_smoothed.stl") == "model/stl"
    assert artifact_mime("topology_smoothed.vtp") == "application/xml"
    assert artifact_mime("weird.bin") == "application/octet-stream"

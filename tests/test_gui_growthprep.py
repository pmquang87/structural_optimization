"""Growth-mesh PREPARE subprocess launcher/attach machinery
(:mod:`oropt.gui.growthprep`).

Launch and classification are hermetic — ``spawn_detached`` is monkeypatched so
nothing real starts, and every state is driven purely through the files the
module itself reads back (pid / log / report.json), which is the whole
contract: the GUI must be able to re-attach from files alone. The
real-subprocess end-to-end sits behind ``pytest.importorskip("tetgen")`` like
the other TetGen-backed tests.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from oropt.config import Config, GrowthBox, LoadCase
from oropt.growthmesh import GROWTH_MESH_DIRNAME, GrowthMeshReport
from oropt.gui import growthprep

from conftest import MINI_DECK

# Same region as test_growthmesh's WING: attached to the part's x=0 face so the
# generated tets are reachable.
WING = GrowthBox(name="wing", x_min=-1.0, x_max=-0.001, y_min=-0.5, y_max=1.5,
                 z_min=-0.5, z_max=1.5)


def _mini_cfg(tmp_path, boxes) -> Config:
    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.model.growth_boxes = list(boxes)
    cfg.load_cases = [LoadCase(name="c", stem="mini", sigma_allow=1.0)]
    return cfg


def _report() -> GrowthMeshReport:
    return GrowthMeshReport(
        out_dir="gm", starters=[], engines=[], n_new_nodes=2, n_new_elems=2,
        n_generated=4, per_region=[("wing", 2)], target_edge=1.0,
        max_volume=0.1, quality_min=0.5, quality_median=0.8,
        node_id_range=(6, 7), elem_id_range=(3, 4), total_candidates=2,
        written=True, original_elem_max=2)


def _dead_pid() -> int:
    """A pid guaranteed to belong to no live process (spawned and reaped)."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class _FakeProc:
    pid = 4242


# ---- prepare_dir --------------------------------------------------------------
def test_prepare_dir_follows_run_folder_model(tmp_path):
    cfg = _mini_cfg(tmp_path, [WING])
    assert growthprep.prepare_dir(cfg) == tmp_path / growthprep.PREPARE_DIRNAME
    cfg.work_dir = str(tmp_path / "work")
    assert (growthprep.prepare_dir(cfg)
            == tmp_path / "work" / growthprep.PREPARE_DIRNAME)


# ---- read_status state machine --------------------------------------------------
def test_status_idle_when_nothing_launched(tmp_path):
    s = growthprep.read_status(tmp_path / "never_created")
    assert (s.state, s.report, s.error) == (growthprep.IDLE, None, "")


def test_status_running_on_live_pid(tmp_path):
    (tmp_path / growthprep.PIDFILE).write_text(str(os.getpid()),
                                               encoding="utf-8")
    (tmp_path / growthprep.LOG_NAME).write_text("line1\nline2\n",
                                                encoding="utf-8")
    s = growthprep.read_status(tmp_path)
    assert s.state == growthprep.RUNNING
    assert s.pid == os.getpid()
    assert s.log_tail.endswith("line2")


def test_status_done_parses_report_back(tmp_path):
    rep = _report()
    (tmp_path / growthprep.PIDFILE).write_text(str(_dead_pid()),
                                               encoding="utf-8")
    (tmp_path / growthprep.REPORT_NAME).write_text(
        json.dumps(dataclasses.asdict(rep)), encoding="utf-8")
    s = growthprep.read_status(tmp_path)
    assert s.state == growthprep.DONE
    assert s.report == rep


def test_status_failed_uses_cli_error_line(tmp_path):
    (tmp_path / growthprep.PIDFILE).write_text(str(_dead_pid()),
                                               encoding="utf-8")
    (tmp_path / growthprep.LOG_NAME).write_text(
        "[oropt] growth-mesh: part surface 4 triangles / 4 nodes; ...\n"
        "[oropt] growth-mesh: ERROR: no growth regions configured\n",
        encoding="utf-8")
    s = growthprep.read_status(tmp_path)
    assert s.state == growthprep.FAILED
    assert s.error == "no growth regions configured"


def test_status_failed_crash_quotes_last_output(tmp_path):
    """A native TetGen abort (or an OOM kill) prints no ERROR line — the exact
    failure mode this module exists to survive."""
    (tmp_path / growthprep.PIDFILE).write_text(str(_dead_pid()),
                                               encoding="utf-8")
    (tmp_path / growthprep.LOG_NAME).write_text(
        "[oropt] growth-mesh: tetrahedralising the PLC (9 points) ...\n",
        encoding="utf-8")
    s = growthprep.read_status(tmp_path)
    assert s.state == growthprep.FAILED
    assert "without writing a report" in s.error
    assert "tetrahedralising" in s.error


def test_status_failed_unreadable_report(tmp_path):
    (tmp_path / growthprep.REPORT_NAME).write_text("{not json",
                                                   encoding="utf-8")
    s = growthprep.read_status(tmp_path)
    assert s.state == growthprep.FAILED
    assert growthprep.REPORT_NAME in s.error


# ---- start ---------------------------------------------------------------------
def test_start_freezes_config_and_launches_cli(tmp_path, monkeypatch):
    seen = {}

    def fake_spawn(cmd, cwd, stdout=None, stderr=None):
        seen["cmd"], seen["cwd"] = list(cmd), Path(cwd)
        return _FakeProc()

    monkeypatch.setattr(growthprep, "spawn_detached", fake_spawn)
    cfg = _mini_cfg(tmp_path, [WING])
    prep = tmp_path / growthprep.PREPARE_DIRNAME
    prep.mkdir()
    # a stale outcome from a previous launch must not survive the relaunch
    (prep / growthprep.REPORT_NAME).write_text("{}", encoding="utf-8")
    pid = growthprep.start(cfg, prep, 1.4, 1.7, tmp_path)
    assert pid == 4242
    assert (prep / growthprep.PIDFILE).read_text(encoding="utf-8") == "4242"
    assert not (prep / growthprep.REPORT_NAME).exists()
    # the frozen config carries the on-screen regions verbatim
    frozen = Config.from_yaml(prep / growthprep.CONFIG_NAME)
    assert frozen.model.growth_boxes == [WING]
    cmd = seen["cmd"]
    assert cmd[:3] == [sys.executable, "-m", "oropt.growthmesh"]
    assert cmd[cmd.index("--config") + 1] == str(prep / growthprep.CONFIG_NAME)
    assert cmd[cmd.index("--size-factor") + 1] == "1.4"
    assert cmd[cmd.index("--min-ratio") + 1] == "1.7"
    assert cmd[cmd.index("--json") + 1] == str(prep / growthprep.REPORT_NAME)
    assert seen["cwd"] == tmp_path


def test_start_refuses_double_launch(tmp_path):
    prep = tmp_path / growthprep.PREPARE_DIRNAME
    prep.mkdir()
    (prep / growthprep.PIDFILE).write_text(str(os.getpid()), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already"):
        growthprep.start(_mini_cfg(tmp_path, [WING]), prep, 1.0, 1.5, tmp_path)


# ---- cancel --------------------------------------------------------------------
def test_cancel_noop_when_nothing_alive(tmp_path):
    growthprep.cancel(tmp_path)                       # no pid file at all
    (tmp_path / growthprep.PIDFILE).write_text(str(_dead_pid()),
                                               encoding="utf-8")
    growthprep.cancel(tmp_path)                       # dead pid — still a no-op


def test_cancel_kills_live_child(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"])
    (tmp_path / growthprep.PIDFILE).write_text(str(proc.pid),
                                               encoding="utf-8")
    try:
        growthprep.cancel(tmp_path)
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode != 0                        # killed, not clean exit


# ---- real subprocess end-to-end (needs tetgen, like the CLI e2e) ----------------
def test_start_to_done_real_subprocess(tmp_path):
    """The full GUI contract against a real detached child: launch, poll via
    files only, parse the report back, decks written where the report says."""
    pytest.importorskip("tetgen")
    import oropt
    (tmp_path / "mini_0000.rad").write_text(MINI_DECK, encoding="utf-8")
    cfg = _mini_cfg(tmp_path, [WING])
    cfg.work_dir = str(tmp_path / "work")
    prep = growthprep.prepare_dir(cfg)
    # cwd = the tree this test imported oropt from, so the child (whose
    # sys.path[0] is its cwd) resolves the same code — what the GUI does with
    # its PROJECT_ROOT.
    root = Path(oropt.__file__).resolve().parents[1]
    growthprep.start(cfg, prep, 1.0, 1.5, root)
    deadline = time.time() + 180
    while growthprep.read_status(prep).state == growthprep.RUNNING:
        assert time.time() < deadline, "PREPARE subprocess never finished"
        time.sleep(0.5)
    s = growthprep.read_status(prep)
    assert s.state == growthprep.DONE, f"{s.error}\n{s.log_tail}"
    assert s.report.n_new_elems > 0
    assert Path(s.report.out_dir) == tmp_path / GROWTH_MESH_DIRNAME
    assert (Path(s.report.out_dir) / "mini_0000.rad").is_file()
    assert "guards passed" in s.log_tail

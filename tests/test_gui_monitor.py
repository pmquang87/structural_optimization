"""Monitor tab: the run.log tail panel that surfaces the loop's progress and —
its reason for existing — every best-effort post-run step's skip reason
(d3plot / smooth / animate / report), which the detached launch would otherwise
discard to DEVNULL. Mirrors the PREPARE panel's log tail."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import oropt
import oropt.gui.runstate as runstate
from oropt import status as st_io
from oropt.config import Config, LoadCase

_APP = Path(oropt.__file__).resolve().parent / "gui" / "app.py"


def _run_app_on(cfg_path: Path):
    # A cold-import AppTest needs a generous timeout (see the apptest-cold-import
    # note); the Monitor follows the selected config's own folder once we stub out
    # "is a run live elsewhere?" so the test never depends on machine state.
    at = AppTest.from_file(str(_APP), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()
    return at


def _make_run(tmp_path, state: str) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    cfg = Config()
    cfg.work_dir = str(work)
    cfg.load_cases = [LoadCase(name="a", stem="gp", sigma_allow=1.0)]
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)
    st_io.write_status(work, st_io.Status(state=state, iteration=2,
                                          volume_fraction=0.8, message="done"))
    (work / st_io.RUN_LOG).write_text(
        "2026-07-07T10:00:00 [oropt] iter 0 done\n"
        "2026-07-07T10:05:00 [oropt] animate: overlay not JSON-serialisable - skipped\n",
        encoding="utf-8")
    return cfg_path


def test_monitor_shows_run_log_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(runstate, "find_active_run", lambda *a, **k: None)
    cfg_path = _make_run(tmp_path, state="failed")

    at = _run_app_on(cfg_path)
    assert not at.exception
    assert any("Run log" in e.label for e in at.expander)          # the panel is there
    assert any("overlay not JSON-serialisable - skipped" in c.value  # the skip reason
               for c in at.code)


def test_monitor_no_run_log_no_panel(tmp_path, monkeypatch):
    """A run with no run.log (e.g. a pre-tee run) shows no empty log panel."""
    monkeypatch.setattr(runstate, "find_active_run", lambda *a, **k: None)
    cfg_path = _make_run(tmp_path, state="stopped")
    (Path(cfg_path).parent / "work" / st_io.RUN_LOG).unlink()      # drop the log

    at = _run_app_on(cfg_path)
    assert not at.exception
    assert not any("Run log" in e.label for e in at.expander)

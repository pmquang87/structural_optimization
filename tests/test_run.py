"""CLI entry point (``python -m oropt.run``): the run.log tee that keeps the
detached run's output — and, crucially, its best-effort post-run steps' skip
reasons — from vanishing into the GUI's DEVNULL launch."""
from __future__ import annotations

import oropt.run as run
from oropt.config import Config
from oropt.status import Status


def test_tee_log_writes_run_log_and_tees_to_stdout(tmp_path, capsys):
    with run._tee_log(tmp_path, resume=False) as log:
        log("[oropt] hello")
        log("[oropt] world")
    text = (tmp_path / run.RUN_LOG).read_text(encoding="utf-8")
    assert "[oropt] hello" in text and "[oropt] world" in text
    assert text.count("\n") == 2                      # one line each
    out = capsys.readouterr().out                     # still prints to stdout
    assert "[oropt] hello" in out and "[oropt] world" in out


def test_tee_log_truncates_fresh_appends_on_resume(tmp_path):
    with run._tee_log(tmp_path, resume=False) as log:
        log("first")
    with run._tee_log(tmp_path, resume=True) as log:  # resume -> append
        log("second")
    text = (tmp_path / run.RUN_LOG).read_text(encoding="utf-8")
    assert "first" in text and "second" in text
    with run._tee_log(tmp_path, resume=False) as log:  # fresh -> truncate
        log("third")
    text = (tmp_path / run.RUN_LOG).read_text(encoding="utf-8")
    assert "third" in text and "first" not in text and "second" not in text


def test_main_persists_post_run_log_line(tmp_path, monkeypatch):
    """A post-run step's log (e.g. an animate skip) lands in run.log even though
    the loop's stdout is discarded when launched detached by the GUI."""
    cfg = Config()
    cfg.work_dir = str(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)

    def fake_run(cfg, resume, log=print):
        log("[oropt] animate: growth-region overlay not JSON-serialisable - skipped")
        return Status(state="stopped", iteration=3, volume_fraction=0.5,
                      message="done")
    monkeypatch.setattr(run, "run_optimization", fake_run)

    rc = run.main(["--config", str(cfg_path), "--skip-validate"])
    assert rc == 0
    text = (tmp_path / run.RUN_LOG).read_text(encoding="utf-8")
    assert "overlay not JSON-serialisable" in text
    assert "finished: state=stopped" in text

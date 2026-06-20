"""Automatic post-run summary report: config plumbing, summary numbers, charts,
best-effort guards. Hermetic — synthesises a run folder from the status helpers
(no solver, no GUI) and never asserts on the GL-dependent topology render."""
from __future__ import annotations

import numpy as np

import oropt.report as report
from oropt import status as st
from oropt.config import Config, ReportOpts
from oropt.report import _render_topology, _summarise, write_report


def _write_topology(work):
    """A tiny topology_latest.vtu: two tets sharing a face, with a sens field."""
    node_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                        dtype=float)
    conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    st.write_topology(work, node_xyz, conn, np.array([True, True]),
                      fields={"sensitivity": np.array([9.0, 8.0])})


def _make_run(work, *, optimizer="beso", state="converged", final_vf=0.5,
              sigma_max=210.5, feasible=True):
    """Synthesise a finished run: topology + 3-row history + final status."""
    _write_topology(work)
    vfs = [1.0, 0.75, final_vf]
    sigs = [200.0, 205.0, sigma_max]
    disps = [0.80, 0.90, 0.95]
    walls = [10.0, 12.0, 11.0]
    for i, (vf, sg, dp, wl) in enumerate(zip(vfs, sigs, disps, walls)):
        st.append_history(work, {
            "iteration": i, "volume_fraction": vf, "sigma_max": sg, "disp": dp,
            "elements_alive": 100 - i, "feasible": feasible,
            "iter_wall_s": wl, "or_termination": "ok"})
    st.write_status(work, st.Status(
        state=state, iteration=2, max_iter=150, volume_fraction=final_vf,
        sigma_max=sigma_max, sigma_allow=250.0, disp=0.95, d_allow=1.0,
        feasible=feasible, elements_alive=98, elements_total=100,
        message="converged at target volume, feasible"))


def _cfg(optimizer="beso"):
    cfg = Config()
    cfg.optimizer = optimizer
    return cfg


# --- config plumbing -------------------------------------------------------- #
def test_report_enabled_by_default_and_roundtrips(tmp_path):
    cfg = Config()
    assert cfg.report.enabled is True                  # cheap + read-only -> on
    cfg.report.render_topology = False
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.report.enabled is True
    assert back.report.render_topology is False


def test_from_dict_ignores_unknown_report_keys():
    cfg = Config.from_dict({"report": {"enabled": False, "bogus": 123}})
    assert cfg.report.enabled is False
    assert isinstance(cfg.report, ReportOpts)


# --- summary numbers -------------------------------------------------------- #
def test_summarise_numbers():
    # _summarise reduces the given status/history to scalars (reads no disk).
    cfg = _cfg("tobs")
    history = [
        {"iteration": 0, "volume_fraction": 1.0, "sigma_max": 200, "disp": 0.8,
         "feasible": True, "iter_wall_s": 10},
        {"iteration": 1, "volume_fraction": 0.5, "sigma_max": 210.5, "disp": 0.95,
         "feasible": True, "iter_wall_s": 12},
    ]
    status = st.Status(state="converged", iteration=1, volume_fraction=0.5,
                       sigma_max=210.5, sigma_allow=250.0, disp=0.95,
                       d_allow=1.0, feasible=True)
    s = _summarise(cfg, status, history)
    assert s.optimizer == "tobs"
    assert s.iterations == 2
    assert abs(s.start_vf - 1.0) < 1e-9 and abs(s.final_vf - 0.5) < 1e-9
    assert abs(s.mass_removed_pct - 50.0) < 1e-9
    assert abs(s.total_wall_s - 22.0) < 1e-9
    assert s.feasible is True and not s.multi_load


# --- end-to-end ------------------------------------------------------------- #
def test_write_report_creates_files_with_key_numbers(tmp_path):
    _make_run(tmp_path)
    cfg = _cfg("beso")
    cfg.report.render_topology = False     # the render is covered separately; keep
    out = write_report(cfg, tmp_path, lambda *_: None)   # this assertion GL-free
    assert out is not None and out.name == "report.html"

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    for text in (html, md):
        assert "beso" in text                          # optimiser name
        assert "50.0%" in text                         # % mass removed
        assert "210.5" in text                         # final sigma
        assert "250.0" in text                         # sigma_allow
        assert "topology_latest.vtu" in text           # artefact link (render-free)
    assert "FEASIBLE" in html


def test_write_report_charts_written(tmp_path):
    # matplotlib is a core dependency, so the convergence charts must be produced.
    _make_run(tmp_path)
    cfg = _cfg("beso")
    cfg.report.render_topology = False     # exercise charts only (no GL needed)
    write_report(cfg, tmp_path, lambda *_: None)
    assert (tmp_path / "report_volume_fraction.png").is_file()
    assert (tmp_path / "report_sigma.png").is_file()
    assert (tmp_path / "report_disp.png").is_file()


def test_write_report_disabled_is_noop(tmp_path):
    _make_run(tmp_path)
    cfg = _cfg("beso")
    cfg.report.enabled = False
    assert write_report(cfg, tmp_path, lambda *_: None) is None
    assert not (tmp_path / "report.html").exists()


def test_write_report_no_data_returns_none(tmp_path):
    logs: list[str] = []
    assert write_report(_cfg("beso"), tmp_path, logs.append) is None
    assert any("no status" in m.lower() for m in logs)


def test_write_report_render_disabled_links_topology(tmp_path):
    # With the off-screen render off we exercise the deterministic link fallback:
    # no PNG, but the topology file is still linked from the report.
    _make_run(tmp_path)
    cfg = _cfg("beso")
    cfg.report.render_topology = False
    write_report(cfg, tmp_path, lambda *_: None)
    assert not (tmp_path / "report_topology.png").exists()
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "topology_latest.vtu" in html


def test_write_report_multiload_worstcase_note(tmp_path):
    _make_run(tmp_path)
    # Two load cases -> the report flags σ/disp as worst-case across cases.
    cfg = Config.from_dict({
        "optimizer": "beso",
        "load_cases": [{"name": "pull"}, {"name": "push"}]})
    cfg.report.render_topology = False
    write_report(cfg, tmp_path, lambda *_: None)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "worst case across 2 load cases" in html
    assert "worst case across 2 load cases" in md


def test_write_report_infeasible_badge(tmp_path):
    _make_run(tmp_path, state="failed", feasible=False, sigma_max=300.0)
    cfg = _cfg("beso")
    cfg.report.render_topology = False
    write_report(cfg, tmp_path, lambda *_: None)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "INFEASIBLE" in html


# --- topology render (isolated subprocess) ---------------------------------- #
def test_render_topology_is_contained(tmp_path):
    # The real off-screen render runs in an isolated subprocess: where a GL
    # context exists it yields the PNG, and on a headless box (e.g. CI) the
    # subprocess fails/crashes and we get None — but never an exception here.
    _write_topology(tmp_path)
    out = _render_topology(tmp_path, 120.0, lambda *_: None)
    assert out is None or out.is_file()


def test_render_failure_falls_back_to_link(tmp_path, monkeypatch):
    # Force the render subprocess to exit non-zero without producing a PNG (stands
    # in for a GL/driver crash): the report must still be written and link the
    # topology files instead of embedding an image.
    _make_run(tmp_path)
    monkeypatch.setattr(report, "_RENDER_RUNNER", "import sys; sys.exit(3)")
    logs: list[str] = []
    write_report(_cfg("beso"), tmp_path, logs.append)
    assert not (tmp_path / "report_topology.png").exists()
    assert any("render failed" in m for m in logs)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "topology_latest.vtu" in html

"""Automatic post-run summary report: config plumbing, summary numbers, charts,
best-effort guards. Hermetic — synthesises a run folder from the status helpers
(no solver, no GUI) and never asserts on the GL-dependent topology render."""
from __future__ import annotations

import numpy as np
import pytest

import oropt.report as report
from oropt import status as st
from oropt.config import Config, ReportOpts
from oropt.report import _render_topology, _summarise, write_report


@pytest.fixture(autouse=True)
def _no_interactive_scene(monkeypatch):
    """Keep the suite hermetic/fast: the report's interactive VTK.js export needs
    the optional trame backend and spawns a subprocess, so default it 'off' for
    every test (regardless of whether trame happens to be installed). The tests
    that exercise it re-enable the backend via monkeypatch."""
    monkeypatch.setattr(report, "_scene_backend_available", lambda: False)


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


# --- reporting the last feasible design (not the last iteration) ------------ #
def _make_mixed_run(work, feas, sigmas, *, sigma_allow=860.0, d_allow=5.0):
    """A finished run whose iterations mix feasible/infeasible, WITH a matching
    per-iteration topology snapshot for each; status.json reflects the LAST
    iteration (as a real run does). ``feas``/``sigmas`` are per-iteration lists."""
    node_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                        dtype=float)
    conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    n = len(feas)
    for i in range(n):
        st.write_topology(work, node_xyz, conn, np.array([True, True]),
                          fields={"sensitivity": np.array([9.0, 8.0])}, iteration=i)
        st.append_history(work, {
            "iteration": i, "volume_fraction": round(1.0 - i * 0.02, 6),
            "sigma_max": sigmas[i], "disp": round(0.9 + i * 0.02, 6),
            "elements_alive": 100 - i, "feasible": feas[i],
            "iter_wall_s": 10.0, "or_termination": "NORMAL TERMINATION",
            "optimizer": "beso"})
    last = n - 1
    st.write_status(work, st.Status(
        state="converged", iteration=last, max_iter=n,
        volume_fraction=1.0 - last * 0.02, sigma_max=sigmas[last],
        sigma_allow=sigma_allow, disp=0.9 + last * 0.02, d_allow=d_allow,
        feasible=feas[last], elements_alive=100 - last, elements_total=100,
        message=("feasible" if feas[last] else "INFEASIBLE - backing off")))


def test_summarise_picks_last_feasible_iteration():
    # 0,1 feasible; 2 (last) infeasible -> the summary describes iteration 1, and
    # records that the run actually ended (infeasibly) at iteration 2.
    cfg = _cfg("beso")
    history = [
        {"iteration": 0, "volume_fraction": 1.0, "sigma_max": 700.0, "disp": 0.9,
         "feasible": True, "iter_wall_s": 10},
        {"iteration": 1, "volume_fraction": 0.96, "sigma_max": 731.5, "disp": 0.94,
         "feasible": True, "iter_wall_s": 10},
        {"iteration": 2, "volume_fraction": 0.94, "sigma_max": 863.9, "disp": 0.98,
         "feasible": False, "iter_wall_s": 10},
    ]
    status = st.Status(state="converged", iteration=2, volume_fraction=0.94,
                       sigma_max=863.9, sigma_allow=860.0, disp=0.98, d_allow=5.0,
                       feasible=False)
    s = _summarise(cfg, status, history)
    assert s.reported_iteration == 1 and s.last_iteration == 2
    assert s.feasible is True and not s.all_infeasible and s.ended_mid_oscillation
    assert abs(s.sigma_max - 731.5) < 1e-6           # reported (iter 1), not 863.9
    assert abs(s.final_vf - 0.96) < 1e-6
    assert abs(s.last_sigma_max - 863.9) < 1e-6      # the run's actual last iteration
    assert abs(s.sigma_allow - 860.0) < 1e-6         # limit still from status


def test_reported_topology_src_prefers_snapshot(tmp_path):
    _make_mixed_run(tmp_path, feas=[True, True, False], sigmas=[700, 731.5, 863.9])
    # the reported (feasible) iteration's snapshot, NOT topology_latest.vtu
    assert report._reported_topology_src(tmp_path, 1).name == "topology_iter0001.vtu"
    # missing snapshot -> falls back to the latest file
    assert report._reported_topology_src(tmp_path, 99).name == st.TOPOLOGY


def test_write_report_headlines_last_feasible_design(tmp_path):
    # Mirrors opti_run5_Ti: the last iteration is infeasible but an earlier one is
    # feasible, so the report headlines the feasible design and notes the ending.
    _make_mixed_run(tmp_path, feas=[True, True, False], sigmas=[700, 731.5, 863.9])
    cfg = _cfg("beso")
    cfg.report.render_topology = False
    write_report(cfg, tmp_path, lambda *_: None)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert 'class="badge ok">FEASIBLE' in html         # headline design is feasible
    assert "731.5 / 860.0 MPa" in html                 # Final σ_max = reported iter 1
    for text in (html, md):
        assert "Reported design" in text
        assert "iteration 1" in text                   # the reported design
        assert "ended mid-oscillation at iteration 2" in text


def test_write_report_renders_reported_snapshot(tmp_path, monkeypatch):
    # write_report must hand the reported iteration's snapshot to the renderer,
    # not topology_latest.vtu (the possibly-infeasible last iteration).
    _make_mixed_run(tmp_path, feas=[True, True, False], sigmas=[700, 731.5, 863.9])
    captured = {}

    def spy(work, timeout, log, boxes_spec=None, src=None):
        captured["src"] = src
        return None

    monkeypatch.setattr(report, "_render_topology", spy)
    write_report(_cfg("beso"), tmp_path, lambda *_: None)
    assert captured["src"].name == "topology_iter0001.vtu"


def test_write_report_all_infeasible_falls_back_to_last(tmp_path):
    # No feasible iteration -> report the last one, clearly labelled infeasible.
    _make_mixed_run(tmp_path, feas=[False, False, False], sigmas=[900, 905, 903])
    cfg = _cfg("beso")
    cfg.report.render_topology = False
    write_report(cfg, tmp_path, lambda *_: None)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert 'class="badge bad">INFEASIBLE' in html
    assert "No iteration satisfied" in html
    assert report._reported_topology_src(tmp_path, 2).name == "topology_iter0002.vtu"


# --- interactive convergence charts (report.html only) ---------------------- #
def test_write_report_interactive_charts_embedded(tmp_path):
    # report.html gets self-contained, offline hover-interactive charts (data + JS
    # inlined), while report.md and the <noscript> fallback keep the static PNGs.
    _make_run(tmp_path)
    cfg = _cfg("beso")
    cfg.report.render_topology = False
    write_report(cfg, tmp_path, lambda *_: None)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    for key in ("ichart-vf", "ichart-sigma", "ichart-disp"):
        assert f'id="{key}"' in html                   # a container per series
    assert "<script>" in html and "addEventListener" in html
    assert "mousemove" in html and "ichart-tip" in html   # hover tooltip wiring
    assert '"reported"' in html and '"limitLabel"' in html  # inlined chart data
    assert "210.5" in html                             # a per-iteration value
    assert '<script src' not in html and 'src="http' not in html  # no external loads
    # static PNGs still generated, kept by md and as the noscript fallback
    assert (tmp_path / "report_sigma.png").is_file()
    assert "report_sigma.png" in md
    assert "<noscript>" in html


def test_chart_payload_is_plain_json_no_nan(tmp_path):
    # NaNs must serialise to JSON null (JSON has no NaN) so JSON.parse-free embedding
    # is valid, and '<' is escaped so a value can't break out of the <script> tag.
    import json
    cfg = _cfg("beso")
    history = [
        {"iteration": 0, "volume_fraction": 1.0, "sigma_max": 700.0, "disp": 0.9,
         "feasible": True},
        {"iteration": 1, "volume_fraction": 0.9, "sigma_max": float("nan"),
         "disp": 0.95, "feasible": True},
    ]
    s = _summarise(cfg, None, history)
    payload = report._chart_payload(history, s)
    assert "NaN" not in payload and "<" not in payload
    data = json.loads(payload)
    sigma = next(sr for sr in data["series"] if sr["key"] == "sigma")
    assert sigma["values"] == [700.0, None]            # NaN -> null


# --- evolution animation in the report -------------------------------------- #
def _write_anim_gif(work, data=b"GIF89a-fake-bytes"):
    (work / report.ANIM_GIF).write_bytes(data)


def test_write_report_embeds_animation_gif(tmp_path):
    # When the evolution GIF exists it's embedded inline (report.html stays a
    # single self-contained file), linked under Artefacts, and referenced by md.
    _make_run(tmp_path)
    _write_anim_gif(tmp_path)
    cfg = _cfg("beso")
    cfg.report.render_topology = False
    write_report(cfg, tmp_path, lambda *_: None)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "<h2>Evolution</h2>" in html
    assert "data:image/gif;base64," in html            # embedded inline
    assert "Evolution animation" in html               # artefact link
    assert "topology_evolution.gif" in md              # md references the sibling


def test_write_report_large_gif_is_linked_not_embedded(tmp_path, monkeypatch):
    # A GIF over the inline cap is linked as the sibling file so report.html
    # doesn't balloon with megabytes of base64.
    _make_run(tmp_path)
    _write_anim_gif(tmp_path)
    monkeypatch.setattr(report, "MAX_INLINE_GIF_BYTES", 1)   # force the link path
    cfg = _cfg("beso")
    cfg.report.render_topology = False
    write_report(cfg, tmp_path, lambda *_: None)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "data:image/gif" not in html                      # not embedded
    assert 'src="topology_evolution.gif"' in html            # linked instead


def test_write_report_without_animation_shows_note(tmp_path):
    # No GIF -> the Evolution section degrades to a note, never an error.
    _make_run(tmp_path)
    cfg = _cfg("beso")
    cfg.report.render_topology = False
    write_report(cfg, tmp_path, lambda *_: None)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<h2>Evolution</h2>" in html
    assert "no evolution animation" in html


# --- interactive final-design viewer ---------------------------------------- #
def _force_scene_export(monkeypatch, *, body="<html><body>FAKE SCENE</body></html>"):
    """Make the interactive export deterministically 'succeed' without trame: the
    backend check passes and the subprocess just writes a tiny fake scene file."""
    monkeypatch.setattr(report, "_scene_backend_available", lambda: True)
    runner = f"import sys; open(sys.argv[2], 'w', encoding='utf-8').write({body!r})"
    monkeypatch.setattr(report, "_SCENE_RUNNER", runner)


def test_write_report_embeds_interactive_scene(tmp_path, monkeypatch):
    # With the export available, report.html inlines the interactive scene in an
    # <iframe srcdoc> (self-contained), links it under Artefacts, and report.md
    # links the standalone viewer. The static PNG isn't produced when the scene is.
    _make_run(tmp_path)
    _force_scene_export(monkeypatch)
    write_report(_cfg("beso"), tmp_path, lambda *_: None)
    assert (tmp_path / "report_topology.html").is_file()
    assert not (tmp_path / "report_topology.png").exists()   # scene wins; no PNG
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert '<iframe class="scene"' in html
    assert "srcdoc=" in html and "FAKE SCENE" in html        # inlined
    assert "Interactive final design" in html                # artefact link
    assert "report_topology.html" in md                      # md links the viewer


def test_write_report_large_scene_is_linked_not_inlined(tmp_path, monkeypatch):
    # A scene over the inline cap is referenced as the sibling file (src=) so
    # report.html stays light instead of inlining megabytes of vtk.js.
    _make_run(tmp_path)
    _force_scene_export(monkeypatch)
    monkeypatch.setattr(report, "MAX_INLINE_SCENE_BYTES", 1)
    write_report(_cfg("beso"), tmp_path, lambda *_: None)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "srcdoc=" not in html
    assert 'src="report_topology.html"' in html


def test_write_report_no_backend_falls_back_to_static(tmp_path, monkeypatch):
    # Without the trame backend the interactive export is skipped (with a hint) and
    # the report uses the static PNG path — here forced to fail, so it links the
    # topology files instead. Never errors, and writes no scene file.
    _make_run(tmp_path)
    monkeypatch.setattr(report, "_scene_backend_available", lambda: False)
    monkeypatch.setattr(report, "_RENDER_RUNNER", "import sys; sys.exit(3)")
    logs: list[str] = []
    write_report(_cfg("beso"), tmp_path, logs.append)
    assert not (tmp_path / "report_topology.html").exists()
    assert any("report3d" in m for m in logs)                # logged the install hint
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "topology_latest.vtu" in html                     # linked fallback


def test_interactive_topology_flag_roundtrips(tmp_path):
    cfg = Config()
    assert cfg.report.interactive_topology is True            # on by default
    cfg.report.interactive_topology = False
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    assert Config.from_yaml(p).report.interactive_topology is False


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

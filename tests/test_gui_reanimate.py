"""The dashboard's 🛠 Re-postprocessing tab: a smoke test that the script renders
the tab end-to-end, that pointing it at a folder with enough per-iteration
snapshots enables the Generate button, and that the Re-generate report button is
gated on a valid run folder (no real GL render is triggered)."""
from __future__ import annotations

from pathlib import Path

import pytest

import oropt
from oropt.config import Config

pytestmark = pytest.mark.gui   # Streamlit AppTest / pandas-pyarrow: excluded on non-Windows CI


def _app_file() -> str:
    return str(Path(oropt.__file__).resolve().parent / "gui" / "app.py")


def _write_cfg(tmp_path) -> Path:
    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.work_dir = str(tmp_path / "work")
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)
    return cfg_path


def test_app_renders_repostprocess_tab(tmp_path):
    """The Streamlit script renders with the Re-postprocessing tab present."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    _write_cfg(tmp_path)
    # Cold-process import of streamlit/pyvista is slow on first run (>3s).
    at = AppTest.from_file(_app_file(), default_timeout=30)
    at.run()
    assert not at.exception
    assert any("Re-postprocessing" in t.label for t in at.tabs)


def test_reanimate_tab_detects_frames_and_enables_generate(tmp_path):
    """Pointing the tab at a folder with ≥2 snapshots reports the frame count and
    enables 🎬 Generate animation; an empty folder leaves it disabled."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    cfg_path = _write_cfg(tmp_path)

    run_dir = tmp_path / "oldrun"
    run_dir.mkdir()
    for it in range(3):                       # smoothed per-iteration surfaces
        (run_dir / f"topology_smoothed_iter{it:04d}.stl").write_bytes(b"")

    at = AppTest.from_file(_app_file(), default_timeout=30)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()   # use our config

    # An empty / nonexistent folder -> no frames -> Generate stays disabled. (Set
    # the field explicitly rather than trust the default, which seeds from whatever
    # run is live on the machine — keeps the test hermetic.)
    folder = next(t for t in at.text_input if t.label == "Run folder")
    folder.set_value(str(tmp_path / "empty")).run()
    gen = next(b for b in at.button if b.label == "🎬 Generate animation")
    assert gen.disabled

    # Pointing at the run with 3 snapshots reports them and enables Generate.
    folder = next(t for t in at.text_input if t.label == "Run folder")
    folder.set_value(str(run_dir)).run()
    assert not at.exception
    assert any("3 frames found" in m.value for m in at.success)
    gen = next(b for b in at.button if b.label == "🎬 Generate animation")
    assert not gen.disabled                   # ≥2 frames + valid .gif name -> enabled


def test_repostprocess_tab_report_button_gated_on_folder(tmp_path):
    """The Re-postprocessing tab exposes a 📝 Re-generate report button, disabled
    for a nonexistent run folder and enabled once it points at an existing one."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    cfg_path = _write_cfg(tmp_path)
    run_dir = tmp_path / "oldrun"
    run_dir.mkdir()

    at = AppTest.from_file(_app_file(), default_timeout=30)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()

    folder = next(t for t in at.text_input if t.label == "Run folder")
    folder.set_value(str(tmp_path / "empty")).run()
    btn = next(b for b in at.button if b.label == "📝 Re-generate report")
    assert btn.disabled                          # nonexistent folder -> disabled

    folder = next(t for t in at.text_input if t.label == "Run folder")
    folder.set_value(str(run_dir)).run()
    assert not at.exception
    btn = next(b for b in at.button if b.label == "📝 Re-generate report")
    assert not btn.disabled                      # existing folder -> enabled


def test_reanimate_surface_colour_is_a_dropdown_with_named_colours(tmp_path):
    """The surface-colour box is now a dropdown of known-good names (+ an Other…
    escape hatch), seeded from the config, so a typo can't reach the renderer."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    cfg_path = _write_cfg(tmp_path)
    at = AppTest.from_file(_app_file(), default_timeout=30)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()

    sb = next(s for s in at.selectbox if s.label == "Surface colour")
    assert "steelblue" in sb.options
    assert any("Other" in o for o in sb.options)
    assert sb.value == "gray"                 # seeded from the AnimateOpts default

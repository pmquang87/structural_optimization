"""Post-run topology-evolution GIF: config plumbing, frame selection, guards and
the Pillow encode. Hermetic — never invokes the GL-dependent off-screen render
(its frames are synthesised as plain PNGs), mirroring test_report.py."""
from __future__ import annotations

import oropt.animate as anim
from oropt.animate import (
    ANIM_GIF, VIEWS, _encode_gif, _frame_sources, _label_for, _resolve_camera,
    _resolve_view, frame_count, main, make_animation, selectable_views)
from oropt.config import AnimateOpts, Config, CustomView


def _touch(work, name):
    (work / name).write_bytes(b"")          # _frame_sources only globs by name
    return work / name


# --- config plumbing -------------------------------------------------------- #
def test_animate_enabled_by_default_and_roundtrips(tmp_path):
    cfg = Config()
    assert cfg.animate.enabled is True
    cfg.animate.fps = 8.0
    cfg.animate.show_labels = False
    cfg.animate.view = "top"
    cfg.animate.azimuth = 45.0
    cfg.animate.elevation = -20.0
    cfg.animate.opacity = 0.4
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.animate.enabled is True
    assert back.animate.fps == 8.0
    assert back.animate.show_labels is False
    assert back.animate.view == "top"
    assert back.animate.azimuth == 45.0
    assert back.animate.elevation == -20.0
    assert back.animate.opacity == 0.4


# --- camera angle ----------------------------------------------------------- #
def test_views_expose_the_documented_presets():
    assert set(VIEWS) == {
        "iso", "front", "back", "left", "right", "top", "bottom"}


def test_resolve_view_maps_presets_to_plotter_methods():
    assert _resolve_view("iso", lambda *_: None) == ("view_isometric", False)
    assert _resolve_view("top", lambda *_: None) == ("view_xy", False)
    assert _resolve_view("bottom", lambda *_: None) == ("view_xy", True)
    assert _resolve_view("FRONT", lambda *_: None) == ("view_xz", False)  # case-insensitive


def test_resolve_view_unknown_falls_back_to_iso_and_logs():
    logs: list[str] = []
    assert _resolve_view("banana", logs.append) == ("view_isometric", False)
    assert any("unknown view" in m for m in logs)


# --- user-defined (custom) camera angles ------------------------------------ #
def test_custom_views_coerced_from_dicts_and_roundtrip(tmp_path):
    cfg = Config()
    cfg.animate.custom_views = [CustomView("tq", "front", 30.0, 10.0)]
    cfg.animate.view = "tq"
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert len(back.animate.custom_views) == 1
    cv = back.animate.custom_views[0]
    assert isinstance(cv, CustomView)              # dict from YAML coerced back
    assert (cv.name, cv.base, cv.azimuth, cv.elevation) == ("tq", "front", 30.0, 10.0)


def test_selectable_views_lists_custom_then_builtins():
    opts = AnimateOpts(custom_views=[CustomView("tq", "front", 30.0, 10.0)])
    sel = selectable_views(opts)
    assert sel[0] == "tq"
    assert set(sel[1:]) == set(VIEWS)


def test_resolve_camera_custom_view_combines_base_and_global_nudge():
    opts = AnimateOpts(view="tq", azimuth=5.0, elevation=1.0,
                       custom_views=[CustomView("tq", "front", 30.0, 10.0)])
    method, negative, az, el = _resolve_camera(opts, lambda *_: None)
    assert (method, negative) == ("view_xz", False)   # base preset 'front'
    assert az == 35.0 and el == 11.0                  # custom offset + global nudge


def test_resolve_camera_builtin_view_uses_only_global_offsets():
    opts = AnimateOpts(view="top", azimuth=15.0, elevation=0.0)
    method, negative, az, el = _resolve_camera(opts, lambda *_: None)
    assert (method, negative, az, el) == ("view_xy", False, 15.0, 0.0)


# --- frame selection -------------------------------------------------------- #
def test_frame_sources_prefers_smoothed_stl_over_raw_vtu(tmp_path):
    for it in range(3):
        _touch(tmp_path, f"topology_smoothed_iter{it:04d}.stl")
        _touch(tmp_path, f"topology_iter{it:04d}.vtu")
    frames = _frame_sources(tmp_path)
    assert [f.name for f in frames] == [
        "topology_smoothed_iter0000.stl",
        "topology_smoothed_iter0001.stl",
        "topology_smoothed_iter0002.stl"]


def test_frame_sources_falls_back_to_raw_vtu(tmp_path):
    for it in range(2):                       # only raw snapshots present
        _touch(tmp_path, f"topology_iter{it:04d}.vtu")
    frames = _frame_sources(tmp_path)
    assert [f.suffix for f in frames] == [".vtu", ".vtu"]


def test_frame_sources_empty_when_nothing(tmp_path):
    assert _frame_sources(tmp_path) == []


def test_frame_count_reports_number_and_kind(tmp_path):
    assert frame_count(tmp_path) == (0, "")          # nothing yet
    for it in range(3):
        _touch(tmp_path, f"topology_smoothed_iter{it:04d}.stl")
    assert frame_count(tmp_path) == (3, "stl")       # smoothed surfaces preferred
    _touch(tmp_path, "extra.vtu")                    # not a frame source -> ignored
    assert frame_count(tmp_path) == (3, "stl")


def test_label_for_parses_iteration():
    assert _label_for(_P("topology_smoothed_iter0007.stl")) == "iter 7"
    assert _label_for(_P("topology_iter0042.vtu")) == "iter 42"
    assert _label_for(_P("topology_latest.vtu")) == ""


class _P:
    """Minimal stand-in exposing only the ``.stem`` _label_for reads."""
    def __init__(self, name):
        self.stem = name.rsplit(".", 1)[0]


# --- guards ----------------------------------------------------------------- #
def test_make_animation_disabled_is_noop(tmp_path):
    cfg = Config()
    cfg.animate.enabled = False
    assert make_animation(cfg, tmp_path, lambda *_: None) is None


def test_make_animation_too_few_frames_skips(tmp_path):
    _touch(tmp_path, "topology_iter0000.vtu")     # a single snapshot is not enough
    logs: list[str] = []
    assert make_animation(Config(), tmp_path, logs.append) is None
    assert any("need >=2" in m for m in logs)


# --- Pillow encode (hermetic, no GL) ---------------------------------------- #
def test_encode_gif_writes_valid_multiframe_gif(tmp_path):
    from PIL import Image
    pngs = []
    for i, shade in enumerate((40, 120, 200)):
        p = tmp_path / f"frame_{i:04d}.png"
        Image.new("RGB", (16, 16), (shade, shade, shade)).save(p)
        pngs.append(p)
    dest = tmp_path / ANIM_GIF
    out = _encode_gif(pngs, dest, AnimateOpts(), lambda *_: None)
    assert out == dest and dest.is_file()
    gif = Image.open(dest)
    assert gif.format == "GIF"
    assert getattr(gif, "n_frames", 1) == 3


# --- re-animate: custom output name (GL-free via a faked frame render) ------- #
def _fake_render_frames(monkeypatch):
    """Patch the GL-dependent frame render to synthesise plain PNGs instead, so
    the rest of make_animation (frame globbing, encode, naming) runs hermetically."""
    from PIL import Image

    def fake(frames, opts, tmp, log, boxes=None):
        pngs = []
        for i in range(len(frames)):
            p = tmp / f"frame_{i:04d}.png"
            Image.new("RGB", (8, 8), (i * 40, i * 40, i * 40)).save(p)
            pngs.append(p)
        return pngs
    monkeypatch.setattr(anim, "_render_frames", fake)


def test_make_animation_writes_custom_out_name(tmp_path, monkeypatch):
    _touch(tmp_path, "topology_iter0000.vtu")        # >=2 frame sources so it renders
    _touch(tmp_path, "topology_iter0001.vtu")
    _fake_render_frames(monkeypatch)
    out = make_animation(Config(), tmp_path, lambda *_: None,
                         out_name="topology_evolution_reanim.gif")
    assert out == tmp_path / "topology_evolution_reanim.gif" and out.is_file()
    assert not (tmp_path / ANIM_GIF).exists()        # the run's original is untouched


def test_render_frames_includes_growth_boxes_in_spec(tmp_path, monkeypatch):
    """Growth regions reach the render spec as overlay primitives (drawn as fixed
    wireframe outlines over every frame). The GL render itself is faked."""
    import json
    from pathlib import Path

    from PIL import Image

    from oropt._render import RenderResult
    from oropt.config import GrowthBox
    from oropt.mesh import overlay_primitives

    captured: dict = {}

    def fake_run_render(script, args, timeout):
        spec = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        captured["spec"] = spec
        for pstr in spec["pngs"]:
            Image.new("RGB", (8, 8), (0, 0, 0)).save(pstr)
        return RenderResult(True, 0, "")
    monkeypatch.setattr(anim, "run_render", fake_run_render)

    frames = [_touch(tmp_path, f"topology_iter000{i}.vtu") for i in range(2)]
    boxes = overlay_primitives([GrowthBox(
        name="b", shape="box", x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
        z_min=0.0, z_max=1.0)])
    out = anim._render_frames(frames, AnimateOpts(), tmp_path, lambda *_: None,
                              boxes=boxes)
    assert out and len(out) == 2
    assert captured["spec"]["boxes"] == boxes
    assert captured["spec"]["boxes"][0]["kind"] == "box"


def test_render_frames_drops_unserialisable_overlay_but_keeps_frames(
        tmp_path, monkeypatch):
    """Regression: an overlay that won't JSON-serialise (numpy hull indices used
    to slip through) must not sink the GIF. The box spec is dropped, the frames
    still render, and the reason is logged loudly."""
    import json
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from oropt._render import RenderResult

    captured: dict = {}

    def fake_run_render(script, args, timeout):
        captured["spec"] = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        for pstr in captured["spec"]["pngs"]:
            Image.new("RGB", (8, 8), (0, 0, 0)).save(pstr)
        return RenderResult(True, 0, "")
    monkeypatch.setattr(anim, "run_render", fake_run_render)

    frames = [_touch(tmp_path, f"topology_iter000{i}.vtu") for i in range(2)]
    bad_boxes = [{"kind": "box", "name": "b", "corners": [[0.0, 0.0, 0.0]],
                  "edges": [[np.int32(0), np.int32(1)]]}]   # numpy -> not JSON-safe
    logs: list[str] = []
    out = anim._render_frames(frames, AnimateOpts(), tmp_path, logs.append,
                              boxes=bad_boxes)
    assert out and len(out) == 2                     # frames still rendered
    assert captured["spec"]["boxes"] == []           # overlay dropped, not fatal
    assert any("overlay not JSON-serialisable" in m for m in logs)


def test_cli_out_flag_passes_through(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_make(cfg, work, log, *, out_name=ANIM_GIF):
        captured.update(out_name=out_name, fps=cfg.animate.fps,
                        w=cfg.animate.window_w, bg=cfg.animate.background)
        return work / out_name
    monkeypatch.setattr(anim, "make_animation", fake_make)
    rc = main([str(tmp_path), "--out", "my.gif", "--fps", "10",
               "--window-w", "640", "--background", "black"])
    assert rc == 0
    assert captured == {"out_name": "my.gif", "fps": 10.0, "w": 640, "bg": "black"}

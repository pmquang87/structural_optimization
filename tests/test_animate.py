"""Post-run topology-evolution GIF: config plumbing, frame selection, guards and
the Pillow encode. Hermetic — never invokes the GL-dependent off-screen render
(its frames are synthesised as plain PNGs), mirroring test_report.py."""
from __future__ import annotations

from oropt.animate import (
    ANIM_GIF, VIEWS, _encode_gif, _frame_sources, _label_for, _resolve_camera,
    _resolve_view, make_animation, selectable_views)
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

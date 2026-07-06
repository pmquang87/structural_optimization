"""Best-effort post-run animation of the topology evolution.

After a run finishes, the loop calls :func:`make_animation` to turn the
per-iteration *smoothed* surfaces into a single GIF, ``topology_evolution.gif``,
in the run folder — a quick visual of material being removed across the
optimisation, ready to drop into a slide or e-mail.

Frame source (auto, first that exists):

* ``topology_smoothed_iterNNNN.stl`` / ``.vtp`` — the per-iteration smoothed
  surfaces produced by :mod:`oropt.smoothing` (the clean, review-ready shapes);
* ``topology_iterNNNN.vtu`` — the raw per-iteration snapshots, used as a fallback
  when smoothing is disabled so the animation still works.

Every frame is rendered from a **single fixed camera** (framed once on the union
of all snapshots' bounding boxes) so the part appears to lose material *in place*
instead of rescaling frame to frame. The viewpoint is the user's choice — a named
preset (``iso`` / ``front`` / ``back`` / ``left`` / ``right`` / ``top`` /
``bottom``) plus optional ``azimuth`` / ``elevation`` offsets in degrees, so any
angle is reachable — but it stays the same for the whole clip. Rendering is done in an *isolated off-screen
pyvista subprocess* — exactly like the report's topology render — so a hard
VTK/OpenGL crash on a headless machine is contained and can only show up here as a
non-zero exit code; we then log and skip. The PNG frames are encoded into the GIF
with Pillow in-process. Nothing here raises: every failure path is caught and
returns ``None`` so an animation problem can never abort or fail a run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from ._render import run_render
from .config import AnimateOpts, Config
from .mesh import overlay_primitives

ANIM_GIF = "topology_evolution.gif"

# Glob patterns for the frame source, in preference order: the per-iteration
# smoothed surfaces first (the clean shapes the user asked about), then the raw
# topology snapshots as a fallback so the animation still works with smoothing off.
_FRAME_GLOBS = (
    "topology_smoothed_iter*.stl",
    "topology_smoothed_iter*.vtp",
    "topology_iter*.vtu",
)
_ITER_RE = re.compile(r"iter(\d+)", re.IGNORECASE)

# Named camera presets -> (pyvista Plotter view method, negative-side flag). The
# *parent* resolves the friendly name to a method here so the subprocess runner
# stays a dumb applier (no mapping to keep in sync). "iso" maps to
# ``view_isometric`` (which takes no ``negative`` kwarg — the runner handles that).
_VIEW_METHODS: dict[str, tuple[str, bool]] = {
    "iso": ("view_isometric", False),
    "front": ("view_xz", False), "back": ("view_xz", True),
    "right": ("view_yz", False), "left": ("view_yz", True),
    "top": ("view_xy", False), "bottom": ("view_xy", True),
}
VIEWS: tuple[str, ...] = tuple(_VIEW_METHODS)   # selectable camera angles


def _resolve_view(name: str, log: Callable[[str], None]) -> tuple[str, bool]:
    """Built-in preset name -> ``(plotter_method, negative)``; unknown -> ``iso``."""
    key = str(name).strip().lower()
    if key not in _VIEW_METHODS:
        log(f"[oropt] animate: unknown view {name!r}; using 'iso' "
            f"(choose from {', '.join(VIEWS)})")
        key = "iso"
    return _VIEW_METHODS[key]


def _custom_views(opts) -> dict:
    """``{lowercased name: CustomView}`` for the configured user-defined angles."""
    out = {}
    for cv in getattr(opts, "custom_views", None) or []:
        name = str(getattr(cv, "name", "")).strip().lower()
        if name:
            out[name] = cv
    return out


def selectable_views(opts) -> list[str]:
    """All angle names the user can pick: the built-in presets + custom names.

    Used by the GUI to populate the camera-angle dropdown. Custom views shadow a
    built-in of the same name (so a user can override ``iso`` if they really want).
    """
    customs = [str(getattr(cv, "name", "")).strip()
               for cv in (getattr(opts, "custom_views", None) or [])]
    customs = [c for c in customs if c]
    seen = {c.lower() for c in customs}
    return customs + [v for v in VIEWS if v not in seen]


def _resolve_camera(opts, log: Callable[[str], None]
                    ) -> tuple[str, bool, float, float]:
    """Selected ``opts.view`` -> ``(plotter_method, negative, azimuth, elevation)``.

    ``view`` may name a built-in preset or one of ``opts.custom_views``. A custom
    view contributes its base preset and its own azimuth/elevation; the global
    ``opts.azimuth`` / ``opts.elevation`` are always added on top as a final nudge.
    """
    name = str(opts.view).strip().lower()
    cv = _custom_views(opts).get(name)
    if cv is not None:
        method, negative = _resolve_view(cv.base, log)
        az = float(cv.azimuth) + float(opts.azimuth)
        el = float(cv.elevation) + float(opts.elevation)
        return method, negative, az, el
    method, negative = _resolve_view(name, log)
    return method, negative, float(opts.azimuth), float(opts.elevation)

# Run by an isolated subprocess (same interpreter, which already has pyvista) so a
# hard VTK/OpenGL crash on a headless box can never bring the run down — it shows
# up here only as a non-zero exit code. argv -> [spec.json] describing the frames,
# the PNGs to write, the camera-independent render settings; the camera is fixed
# once from the union of every frame's bounds so the design never rescales.
_RENDER_RUNNER = r"""
import json, sys
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
spec = json.loads(open(sys.argv[1], encoding="utf-8").read())

def _surface(path):
    g = pv.read(path)
    return g if isinstance(g, pv.PolyData) else g.extract_surface()

surfaces = [_surface(p) for p in spec["frames"]]
b = np.array([s.bounds for s in surfaces], dtype=float)   # (n, 6): xmin,xmax,...
union = [b[:, 0].min(), b[:, 1].max(),
         b[:, 2].min(), b[:, 3].max(),
         b[:, 4].min(), b[:, 5].max()]

# Growth-region wireframe outlines, drawn (fixed) over every frame so the user
# can see where material was allowed to grow. Their bounds extend the framing
# union so an outline stays in view even where nothing grew.
_boxes = spec.get("boxes", [])

def _box_bounds(pr):
    if pr["kind"] in ("box", "polyhedron"):
        pb = np.asarray(pr["corners"], dtype=float)
        return pb.min(0), pb.max(0)
    if pr["kind"] == "sphere":
        c = np.asarray(pr["center"], dtype=float); r = float(pr["radius"])
        return c - r, c + r
    a = np.asarray(pr["p1"], dtype=float); z = np.asarray(pr["p2"], dtype=float)
    r = float(pr["radius"])
    return np.minimum(a, z) - r, np.maximum(a, z) + r

for pr in _boxes:
    lo, hi = _box_bounds(pr)
    union = [min(union[0], lo[0]), max(union[1], hi[0]),
             min(union[2], lo[1]), max(union[3], hi[1]),
             min(union[4], lo[2]), max(union[5], hi[2])]

def _add_boxes(p):
    for pr in _boxes:
        k = pr["kind"]
        if k in ("box", "polyhedron"):
            pts = np.asarray(pr["corners"], dtype=float)
            lines = np.hstack([[2, i, j] for i, j in pr["edges"]]).astype(int)
            m = pv.PolyData(pts, lines=lines)
        elif k == "sphere":
            m = pv.Sphere(radius=pr["radius"], center=pr["center"])
        else:
            a = np.asarray(pr["p1"], dtype=float); z = np.asarray(pr["p2"], dtype=float)
            m = pv.Cylinder(center=(a + z) / 2.0, direction=z - a,
                            radius=pr["radius"], height=float(np.linalg.norm(z - a)))
        p.add_mesh(m, color="red", style="wireframe", line_width=2,
                   opacity=0.7, reset_camera=False)

opacity = float(spec.get("opacity", 1.0))
p = pv.Plotter(off_screen=True, window_size=spec["window_size"])
p.background_color = spec["background"]
if opacity < 1.0:
    try:
        p.enable_depth_peeling(10)            # correct order-independent transparency
    except Exception:
        pass                                  # driver without it -> plain blending
# Frame the camera ONCE on the union box (an invisible actor) so every frame
# shares the same view and the shrinking design stays put instead of zooming.
p.add_mesh(pv.Box(union), opacity=0.0)
_view = getattr(p, spec["view_method"])
try:
    _view(negative=spec["view_negative"])     # view_xy/xz/yz take a negative side
except TypeError:
    _view()                                   # view_isometric takes no kwargs
az, el = float(spec.get("azimuth", 0.0)), float(spec.get("elevation", 0.0))
if az or el:
    c = p.camera
    if az:
        c.Azimuth(az)                         # relative rotation about the focal pt
    if el:
        c.Elevation(el)
    c.OrthogonalizeViewUp()
p.reset_camera()                              # refit distance along the chosen angle
cam = p.camera_position
p.clear()

for surf, out, label in zip(surfaces, spec["pngs"], spec["labels"]):
    p.clear()
    p.add_mesh(surf, color=spec["color"], show_edges=spec["show_edges"],
               opacity=opacity, smooth_shading=True, reset_camera=False)
    _add_boxes(p)
    p.camera_position = cam
    if label:
        p.add_text(label, position="upper_left", font_size=12, color="black")
    p.screenshot(out)
p.close()
"""


def _frame_sources(work: Path) -> list[Path]:
    """The per-iteration snapshots to animate, sorted by iteration.

    Returns the first non-empty match of :data:`_FRAME_GLOBS` (smoothed surfaces
    preferred, raw ``.vtu`` snapshots as fallback), so the animation uses the
    clean shapes when smoothing ran and still works when it didn't.
    """
    for pattern in _FRAME_GLOBS:
        hits = sorted(work.glob(pattern))
        if hits:
            return hits
    return []


def frame_count(work: Path) -> tuple[int, str]:
    """``(n_frames, source_kind)`` available to animate in *work*.

    A small public wrapper over :func:`_frame_sources` for the GUI's Re-animate
    tab (and any other caller that wants to *preview* what a re-render would use):
    how many per-iteration snapshots would be animated and their file kind
    (``"stl"`` / ``"vtp"`` / ``"vtu"``, ``""`` when none), so the tool can tell the
    user what it found before spending time on a render.
    """
    frames = _frame_sources(Path(work))
    return len(frames), (frames[0].suffix.lstrip(".") if frames else "")


def _label_for(src: Path) -> str:
    """``topology_smoothed_iter0007.stl`` -> ``"iter 7"`` (empty if no match)."""
    m = _ITER_RE.search(src.stem)
    return f"iter {int(m.group(1))}" if m else ""


def _render_frames(frames: list[Path], opts: AnimateOpts, tmp: Path,
                   log: Callable[[str], None], boxes=None) -> Optional[list[Path]]:
    """Render *frames* to ``<tmp>/frame_NNNN.png`` via the isolated subprocess.

    *boxes* is the growth-region overlay spec (:func:`oropt.mesh.overlay_primitives`);
    each region is drawn as a fixed red wireframe outline over every frame.
    Returns the list of PNG paths (in order) on success, else ``None`` (reason
    logged): there is nothing the run depends on here, so any failure of the
    crash-prone GL render degrades to "no animation" rather than aborting.
    """
    pngs = [tmp / f"frame_{i:04d}.png" for i in range(len(frames))]
    view_method, view_negative, azimuth, elevation = _resolve_camera(opts, log)
    spec = {
        "frames": [str(f) for f in frames],
        "pngs": [str(p) for p in pngs],
        "labels": [_label_for(f) if opts.show_labels else "" for f in frames],
        "window_size": [int(opts.window_w), int(opts.window_h)],
        "color": opts.color,
        "opacity": max(0.0, min(1.0, float(opts.opacity))),
        "background": opts.background,
        "show_edges": bool(opts.show_edges),
        "view_method": view_method,
        "view_negative": view_negative,
        "azimuth": azimuth,
        "elevation": elevation,
        "boxes": boxes or [],
    }
    spec_path = tmp / "anim_spec.json"
    # Pre-validate the spec is JSON-serialisable *before* launching the render.
    # The growth-region overlay is decoration; a non-serialisable box spec (e.g.
    # numpy indices leaking through) must never sink the whole GIF -- drop the
    # overlay and animate the frames anyway, matching the report's
    # overlay-is-best-effort behaviour, and say so loudly. Only if even the
    # frames-only spec won't serialise do we give up.
    try:
        payload = json.dumps(spec)
    except TypeError as exc:
        log(f"[oropt] animate: growth-region overlay not JSON-serialisable "
            f"({exc}); rendering without it")
        spec["boxes"] = []
        try:
            payload = json.dumps(spec)
        except TypeError as exc2:
            log(f"[oropt] animate: render spec not JSON-serialisable "
                f"({exc2}) - skipped")
            return None
    spec_path.write_text(payload, encoding="utf-8")
    result = run_render(_RENDER_RUNNER, [spec_path], float(opts.render_timeout_s))
    if not result.ok:
        if result.returncode is None:                 # timed out / could not launch
            log(f"[oropt] animate: {result.detail} - skipped")
        else:
            log(f"[oropt] animate: off-screen render failed "
                f"(rc={result.returncode}): {result.detail} - skipped")
        return None
    written = [p for p in pngs if p.is_file()]
    if len(written) < 2:
        log("[oropt] animate: renderer produced too few frames - skipped")
        return None
    return written


def _encode_gif(pngs: list[Path], dest: Path, opts: AnimateOpts,
                log: Callable[[str], None]) -> Optional[Path]:
    """Encode the rendered PNG frames into an animated GIF with Pillow."""
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        log(f"[oropt] animate: Pillow unavailable: {exc} - skipped")
        return None
    try:
        imgs = [Image.open(p).convert("RGB") for p in pngs]
        ms = max(1, int(round(1000.0 / float(opts.fps))))
        durations = [ms] * len(imgs)
        durations[-1] = ms * max(1, int(opts.hold_last))   # linger on the result
        imgs[0].save(dest, save_all=True, append_images=imgs[1:],
                     duration=durations, loop=0, optimize=True, disposal=2)
    except Exception as exc:  # noqa: BLE001
        log(f"[oropt] animate: GIF encode failed: {exc} - skipped")
        return None
    return dest


def make_animation(cfg: Config, work: Path,
                   log: Callable[[str], None] = print,
                   *, out_name: str = ANIM_GIF) -> Optional[Path]:
    """If enabled, write the topology-evolution GIF from the per-iteration surfaces.

    Renders the smoothed per-iteration surfaces (or the raw ``.vtu`` snapshots as a
    fallback) from a fixed camera and encodes them into an animated GIF in *work*.
    Returns the GIF path, else ``None`` (reason logged) when disabled, when there
    are fewer than two snapshots, or when rendering/encoding fails. Never raises.

    The GIF is named *out_name* (default ``topology_evolution.gif`` — what the loop
    and the report expect). The GUI's Re-animate tool passes a different name so a
    re-render with fresh settings can sit alongside the run's original instead of
    overwriting it.
    """
    opts = getattr(cfg, "animate", None) or AnimateOpts()
    if not opts.enabled:
        return None
    try:
        work = Path(work)
        frames = _frame_sources(work)
        if len(frames) < 2:
            log("[oropt] animate: need >=2 per-iteration snapshots to animate "
                f"(found {len(frames)}) - skipped")
            return None
        dest = work / out_name
        boxes = overlay_primitives(getattr(cfg.model, "growth_boxes", None))
        with tempfile.TemporaryDirectory(prefix="oropt_anim_", dir=work) as td:
            pngs = _render_frames(frames, opts, Path(td), log, boxes=boxes)
            if not pngs:
                return None
            out = _encode_gif(pngs, dest, opts, log)
        if out is None:
            return None
        log(f"[oropt] animate: wrote {out.name} ({len(frames)} frames @ "
            f"{opts.fps:g} fps, view={opts.view} from "
            f"{frames[0].suffix.lstrip('.')} surfaces)")
        return out
    except Exception as exc:  # noqa: BLE001  (best-effort: never fail the run)
        log(f"[oropt] animate: unexpected error: {exc} - skipped")
        return None


def main(argv=None) -> int:
    """Standalone: ``python -m oropt.animate <run_dir>`` to (re)build the GIF.

    Lets you generate ``topology_evolution.gif`` for any existing run folder
    without re-running the optimisation. Reads only the per-iteration snapshots
    already in the folder.
    """
    ap = argparse.ArgumentParser(
        prog="oropt-animate",
        description="Build topology_evolution.gif from a run folder's "
                    "per-iteration (smoothed) surfaces.")
    ap.add_argument("run_dir", help="run folder containing the per-iteration "
                                    "topology_smoothed_iter*/topology_iter* files")
    ap.add_argument("--out", default=ANIM_GIF,
                    help=f"output GIF name written into the run folder "
                         f"(default: {ANIM_GIF}; pick another to keep the original)")
    ap.add_argument("--fps", type=float, default=None, help="frames per second")
    ap.add_argument("--view", choices=VIEWS, default=None,
                    help="camera angle (default: iso)")
    ap.add_argument("--azimuth", type=float, default=None,
                    help="extra camera azimuth rotation [deg] after the preset")
    ap.add_argument("--elevation", type=float, default=None,
                    help="extra camera elevation rotation [deg] after the preset")
    ap.add_argument("--color", default=None, help="surface colour")
    ap.add_argument("--background", default=None, help="frame background colour")
    ap.add_argument("--opacity", type=float, default=None,
                    help="surface opacity 0..1 (1 = solid, <1 = see-through)")
    ap.add_argument("--window-w", type=int, default=None,
                    help="render width [px] (resolution)")
    ap.add_argument("--window-h", type=int, default=None,
                    help="render height [px] (resolution)")
    ap.add_argument("--hold-last", type=int, default=None,
                    help="linger on the final design (x frame duration)")
    ap.add_argument("--show-edges", action="store_true",
                    help="draw mesh edges on the surface")
    ap.add_argument("--no-labels", action="store_true",
                    help="don't stamp 'iter N' on each frame")
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.animate.enabled = True
    if args.fps is not None:
        cfg.animate.fps = args.fps
    if args.view is not None:
        cfg.animate.view = args.view
    if args.azimuth is not None:
        cfg.animate.azimuth = args.azimuth
    if args.elevation is not None:
        cfg.animate.elevation = args.elevation
    if args.color is not None:
        cfg.animate.color = args.color
    if args.background is not None:
        cfg.animate.background = args.background
    if args.opacity is not None:
        cfg.animate.opacity = args.opacity
    if args.window_w is not None:
        cfg.animate.window_w = args.window_w
    if args.window_h is not None:
        cfg.animate.window_h = args.window_h
    if args.hold_last is not None:
        cfg.animate.hold_last = args.hold_last
    if args.show_edges:
        cfg.animate.show_edges = True
    if args.no_labels:
        cfg.animate.show_labels = False

    out = make_animation(cfg, Path(args.run_dir), log=lambda s: print(s, flush=True),
                         out_name=args.out)
    if out is None:
        print("[oropt] animate: no GIF produced (see messages above)", flush=True)
        return 1
    print(f"[oropt] animate: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

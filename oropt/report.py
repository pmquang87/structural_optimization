"""Best-effort automatic post-run summary report for an oropt run.

After a run finishes, the loop calls :func:`write_report` to write a
human-readable summary of the run into the run folder:

* ``report.html`` — a self-contained page (the convergence charts and a render of
  the final design are embedded inline, so it can be e-mailed/archived as one
  file) with the run summary table, feasibility verdict and artefact links;
* ``report.md``   — the same numbers as lightweight Markdown (charts/render
  referenced as sibling ``report_*.png`` files).

It only *reads* the artefacts the loop already wrote (``status.json``,
``history.csv``, ``topology_latest.vtu``), so it never touches the run. Charts
need matplotlib and the topology render needs an off-screen pyvista; both are
best-effort — a missing/failing dependency is logged and degrades to a file link.

The final design is shown as an **interactive zoom/rotate viewer** (the same
VTK.js scene the GUI's Monitor tab renders) when pyvista's ``Plotter.export_html``
and its optional **trame** export backend (``trame`` / ``trame-vtk``, the
``oropt[report3d]`` extra) are available; otherwise it degrades to the static
off-screen PNG, and then to a plain file link — so the report is identical to
before when the extra isn't installed.

Nothing here raises: every failure path is caught and returns/skips so a report
problem can never abort or fail an optimisation run.
"""
from __future__ import annotations

import base64
import datetime as _dt
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Optional

from . import status as st
from ._render import run_render
from .animate import ANIM_GIF
from .config import Config

REPORT_HTML = "report.html"
REPORT_MD = "report.md"
TOPOLOGY_PNG = "report_topology.png"
TOPOLOGY_SCENE_HTML = "report_topology.html"   # standalone interactive VTK.js scene

# The evolution GIF and the interactive scene are embedded inline so report.html
# stays a single emailable file — but only while each is small enough; above the
# cap we reference the sibling file instead so the HTML doesn't balloon. (The
# VTK.js scene bundles vtk.js, so it is typically a few MB.)
MAX_INLINE_GIF_BYTES = 8 * 1024 * 1024
MAX_INLINE_SCENE_BYTES = 12 * 1024 * 1024

# Run by an isolated subprocess (same interpreter, which already has pyvista) so a
# hard VTK/OpenGL crash on a headless box can never bring the run down — it shows
# up here only as a non-zero exit code. argv -> [vtu_in, png_out].
_RENDER_RUNNER = (
    "import sys, pyvista as pv; "
    "pv.OFF_SCREEN = True; "
    "g = pv.read(sys.argv[1]); "
    "s = 'sensitivity' if 'sensitivity' in g.cell_data else None; "
    "p = pv.Plotter(window_size=[900, 600], off_screen=True); "
    "p.add_mesh(g, scalars=s, cmap='viridis', show_edges=False); "
    "p.view_isometric(); p.background_color = 'white'; "
    "p.screenshot(sys.argv[2]); p.close()"
)
# Same scene as the PNG render, but exported as an *interactive* standalone VTK.js
# page (zoom/rotate) via pyvista's export_html instead of a screenshot. Also run in
# the isolated subprocess so a GL/driver crash stays contained. argv -> [vtu_in,
# html_out]. Needs the optional trame export backend (see _scene_backend_available).
_SCENE_RUNNER = (
    "import sys, pyvista as pv; "
    "pv.OFF_SCREEN = True; "
    "g = pv.read(sys.argv[1]); "
    "s = 'sensitivity' if 'sensitivity' in g.cell_data else None; "
    "p = pv.Plotter(window_size=[900, 600], off_screen=True); "
    "p.add_mesh(g, scalars=s, cmap='viridis', show_edges=False); "
    "p.view_isometric(); p.background_color = 'white'; "
    "p.export_html(sys.argv[2]); p.close()"
)
_CHART_FILES = {
    "vf": "report_volume_fraction.png",
    "sigma": "report_sigma.png",
    "disp": "report_disp.png",
}


# --------------------------------------------------------------------------- #
# numeric helpers
# --------------------------------------------------------------------------- #
def _f(v) -> float:
    """Parse a value (CSV cell, JSON number, ...) to float; NaN on failure."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _isnan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _num(x: float, digits: int = 3) -> str:
    """Fixed-point string, or ``"n/a"`` for NaN/None."""
    if x is None or _isnan(_f(x)):
        return "n/a"
    return f"{float(x):.{digits}f}"


def _pct(x: float, digits: int = 1) -> str:
    return "n/a" if _isnan(_f(x)) else f"{float(x):.{digits}f}%"


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
@dataclass
class Summary:
    optimizer: str
    iterations: int
    state: str
    message: str
    feasible: Optional[bool]
    start_vf: float
    final_vf: float
    mass_removed_pct: float
    sigma_max: float
    sigma_allow: float
    disp: float
    d_allow: float
    total_wall_s: float
    n_cases: int
    stress_excluded: int = 0

    @property
    def multi_load(self) -> bool:
        return self.n_cases > 1


def _summarise(cfg: Config, status: Optional[st.Status],
               history: list[dict]) -> Summary:
    """Reduce ``status.json`` + ``history.csv`` to the report's scalar summary."""
    vfs = [_f(r.get("volume_fraction")) for r in history]
    vfs = [v for v in vfs if not _isnan(v)]
    walls = [_f(r.get("iter_wall_s")) for r in history]
    total_wall = sum(w for w in walls if not _isnan(w))

    start_vf = vfs[0] if vfs else float("nan")
    if status is not None and not _isnan(_f(status.volume_fraction)):
        final_vf = float(status.volume_fraction)
    else:
        final_vf = vfs[-1] if vfs else float("nan")
    if not _isnan(start_vf) and not _isnan(final_vf) and start_vf > 0:
        mass_removed = (start_vf - final_vf) / start_vf * 100.0
    else:
        mass_removed = float("nan")

    last = history[-1] if history else {}

    def _from(attr: str, key: str, fallback: float) -> float:
        if status is not None and not _isnan(_f(getattr(status, attr))):
            return float(getattr(status, attr))
        if key in last:
            return _f(last[key])
        return fallback

    feasible: Optional[bool]
    if status is not None:
        feasible = bool(status.feasible)
    elif "feasible" in last:
        feasible = str(last["feasible"]).strip().lower() in ("true", "1", "yes")
    else:
        feasible = None

    return Summary(
        optimizer=cfg.optimizer_name(),
        iterations=(len(history) if history
                    else (status.iteration + 1 if status else 0)),
        state=(status.state if status else ""),
        message=(status.message if status else ""),
        feasible=feasible,
        start_vf=start_vf,
        final_vf=final_vf,
        mass_removed_pct=mass_removed,
        sigma_max=_from("sigma_max", "sigma_max", float("nan")),
        sigma_allow=_from("sigma_allow", "", cfg.constraints.sigma_allow),
        disp=_from("disp", "disp", float("nan")),
        d_allow=_from("d_allow", "", cfg.constraints.d_allow),
        total_wall_s=total_wall,
        n_cases=len(cfg.load_cases),
        stress_excluded=(int(getattr(status, "stress_excluded_elems", 0))
                         if status is not None else 0),
    )


def _rows(s: Summary) -> list[tuple[str, str]]:
    """(label, value) pairs for the summary table, shared by HTML and Markdown."""
    sigma = f"{_num(s.sigma_max, 1)} / {_num(s.sigma_allow, 1)} MPa"
    disp = f"{_num(s.disp, 4)} / {_num(s.d_allow, 4)} mm"
    if s.feasible is None:
        verdict = "unknown"
    else:
        verdict = "FEASIBLE" if s.feasible else "INFEASIBLE"
    return [
        ("Optimiser", s.optimizer),
        ("Termination", f"{s.state or 'n/a'}"
                        + (f" — {s.message}" if s.message else "")),
        ("Iterations", str(s.iterations)),
        ("Volume fraction", f"{_num(s.start_vf, 3)} → {_num(s.final_vf, 3)}"),
        ("Mass removed", _pct(s.mass_removed_pct, 1)),
        (f"Final σ_max{' (worst case)' if s.multi_load else ''}", sigma),
        (f"Final disp{' (worst case)' if s.multi_load else ''}", disp),
        ("Feasible", verdict),
        ("Total wall time", _hms(s.total_wall_s)),
    ]


def _hms(seconds: float) -> str:
    if _isnan(_f(seconds)):
        return "n/a"
    total = int(round(float(seconds)))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


# --------------------------------------------------------------------------- #
# charts (matplotlib, best-effort)
# --------------------------------------------------------------------------- #
def _charts(history: list[dict], s: Summary, work: Path,
            log: Callable[[str], None]) -> dict[str, Path]:
    """Render the convergence charts to ``report_*.png`` in *work*.

    Returns ``{name: png_path}`` for each chart written (empty if matplotlib is
    unavailable, there is no history, or rendering fails).
    """
    if not history:
        return {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:  # noqa: BLE001
        log(f"[oropt] report: matplotlib unavailable: {exc} - charts skipped")
        return {}

    it = [_f(r.get("iteration")) for r in history]
    vf = [_f(r.get("volume_fraction")) for r in history]
    sig = [_f(r.get("sigma_max")) for r in history]
    dsp = [_f(r.get("disp")) for r in history]

    specs = [
        ("vf", "Volume fraction", vf, "volume fraction", None, None),
        ("sigma", "von-Mises σ_max vs limit", sig, "σ_max [MPa]",
         s.sigma_allow, "σ_allow"),
        ("disp", "Displacement vs limit", dsp, "disp [mm]",
         s.d_allow, "d_allow"),
    ]
    out: dict[str, Path] = {}
    for key, title, ydata, ylabel, limit, limit_label in specs:
        dest = work / _CHART_FILES[key]
        try:
            fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=110)
            ax.plot(it, ydata, marker="o", ms=3, lw=1.4, color="#1f6feb")
            if limit is not None and not _isnan(_f(limit)):
                ax.axhline(float(limit), ls="--", lw=1.2, color="#cf222e",
                           label=limit_label)
                ax.legend(loc="best", fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("iteration", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            # iteration is a count -> integer-only ticks (no "2.5")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(dest)
            plt.close(fig)
            out[key] = dest
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] report: chart {key} failed: {exc} - skipped")
            try:
                plt.close("all")
            except Exception:  # noqa: BLE001
                pass
    return out


# --------------------------------------------------------------------------- #
# final-topology render (off-screen pyvista, best-effort)
# --------------------------------------------------------------------------- #
def _render_topology(work: Path, timeout_s: float,
                     log: Callable[[str], None]) -> Optional[Path]:
    """Off-screen render of the final design to ``report_topology.png``.

    Renders ``topology_latest.vtu`` exactly like the GUI's off-screen pyvista
    view, but in an *isolated subprocess* so a hard VTK/OpenGL crash on a headless
    machine (no GL context) can never abort the run — it just shows up as a
    non-zero exit code and we fall back to linking the topology files. Returns the
    PNG path, or ``None`` (reason logged) when there is nothing to render or the
    render subprocess fails/times out/crashes.
    """
    src = Path(work) / st.TOPOLOGY
    if not src.is_file():
        log(f"[oropt] report: no {st.TOPOLOGY} to render - topology image skipped")
        return None
    dest = Path(work) / TOPOLOGY_PNG
    dest.unlink(missing_ok=True)        # don't mistake a stale PNG for a fresh one
    result = run_render(_RENDER_RUNNER, [src, dest], timeout_s)
    if result.ok and dest.is_file():
        return dest
    if result.timed_out:
        log(f"[oropt] report: topology render timed out after {timeout_s:.0f}s "
            "- linking files")
    elif result.returncode is None:
        log(f"[oropt] report: {result.detail} - linking files")
    else:
        log(f"[oropt] report: off-screen render failed (rc={result.returncode}): "
            f"{result.detail} - linking files")
    return None


def _scene_backend_available() -> bool:
    """Whether ``Plotter.export_html``'s trame backend is importable here.

    The export subprocess uses this same interpreter, so an in-process
    :func:`importlib.util.find_spec` of the two distinct pieces pyvista needs for
    the offline HTML export — ``trame_vtk`` (the vtk.js exporter) and
    ``nest_asyncio2`` (to launch trame's server synchronously) — predicts whether
    the export can work. Both ship with the optional ``report3d`` extra
    (``pyvista[jupyter]``); when either is missing we skip a doomed subprocess and
    fall straight back to the static PNG.
    """
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("trame_vtk", "nest_asyncio2"))


def _export_topology_scene(work: Path, timeout_s: float,
                           log: Callable[[str], None]) -> Optional[Path]:
    """Export an interactive (zoom/rotate) VTK.js scene to ``report_topology.html``.

    Renders the same ``topology_latest.vtu`` scene as :func:`_render_topology`, but
    via pyvista's ``export_html`` into a standalone, offline interactive viewer —
    so the report's final design can be orbited/zoomed like the GUI's Monitor tab.
    Runs in the *isolated subprocess* (crash containment). Returns the scene path,
    or ``None`` (reason logged) when the optional trame backend is missing, there
    is nothing to render, or the export fails/times out/crashes — the caller then
    falls back to the static PNG. Never raises.
    """
    if not _scene_backend_available():
        log("[oropt] report: interactive 3D viewer needs the optional 'report3d' "
            "extra (trame-vtk) - using a static image "
            "(pip install \"oropt[report3d]\")")
        return None
    src = Path(work) / st.TOPOLOGY
    if not src.is_file():
        log(f"[oropt] report: no {st.TOPOLOGY} to render - interactive view skipped")
        return None
    dest = Path(work) / TOPOLOGY_SCENE_HTML
    dest.unlink(missing_ok=True)        # don't mistake a stale scene for a fresh one
    result = run_render(_SCENE_RUNNER, [src, dest], timeout_s)
    if result.ok and dest.is_file():
        return dest
    if result.timed_out:
        log(f"[oropt] report: interactive export timed out after {timeout_s:.0f}s "
            "- falling back to a static image")
    elif result.returncode is None:
        log(f"[oropt] report: {result.detail} - falling back to a static image")
    else:
        log(f"[oropt] report: interactive export failed (rc={result.returncode}): "
            f"{result.detail} - falling back to a static image")
    return None


# --------------------------------------------------------------------------- #
# artefact links
# --------------------------------------------------------------------------- #
def _artefacts(work: Path) -> list[tuple[str, str]]:
    """(label, relative-name) of the run deliverables that actually exist."""
    out: list[tuple[str, str]] = []
    for label, name in (("Final status", st.STATUS),
                        ("Iteration history", st.HISTORY),
                        ("Final topology (VTU)", st.TOPOLOGY)):
        if (work / name).is_file():
            out.append((label, name))
    for sm in sorted(work.glob("topology_smoothed.*")):
        out.append(("Smoothed surface", sm.name))
    snaps = sorted(work.glob("topology_iter*.vtu"))
    if snaps:
        out.append((f"Per-iteration snapshots ({len(snaps)})", snaps[0].name))
    smoothed_snaps = sorted(work.glob("topology_smoothed_iter*.*"))
    if smoothed_snaps:
        out.append((f"Per-iteration smoothed surfaces ({len(smoothed_snaps)})",
                    smoothed_snaps[0].name))
    if (work / TOPOLOGY_SCENE_HTML).is_file():
        out.append(("Interactive final design", TOPOLOGY_SCENE_HTML))
    if (work / ANIM_GIF).is_file():
        out.append(("Evolution animation", ANIM_GIF))
    if (work / "d3plot").is_dir():
        out.append(("LS-Dyna d3plot", "d3plot/"))
    return out


def _data_uri(png: Path, mime: str = "image/png") -> Optional[str]:
    try:
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:{mime};base64,{b64}"


def _animation_html(work: Path) -> str:
    """The ``<figure>`` for the topology-evolution GIF, or a note when absent.

    Inlines the GIF as a data URI while it's under :data:`MAX_INLINE_GIF_BYTES`
    (keeps report.html self-contained); a larger GIF is linked as the sibling file
    so the HTML stays light. The GIF is written by :func:`oropt.animate.make_animation`,
    which the loop now runs *before* the report.
    """
    gif = work / ANIM_GIF
    if not gif.is_file():
        return ('  <p class="note">(no evolution animation — enable '
                '<code>animate</code>, or see the per-iteration files under '
                '<em>Artefacts</em>)</p>')
    try:
        small = gif.stat().st_size <= MAX_INLINE_GIF_BYTES
    except OSError:
        small = False
    src = (_data_uri(gif, "image/gif") if small else None) or escape(ANIM_GIF)
    return (f'  <figure class="anim"><img alt="Topology evolution" src="{src}">'
            f'<figcaption>Topology evolution</figcaption></figure>')


def _final_design_html(scene: Optional[Path], topo: Optional[Path]) -> str:
    """The 'Final design' block: interactive viewer if exported, else the static
    PNG, else a note pointing at the topology files.

    The interactive scene (zoom/rotate, like the Monitor tab) is inlined via an
    ``<iframe srcdoc>`` while it's under :data:`MAX_INLINE_SCENE_BYTES` so
    report.html stays one self-contained, offline-viewable file; a larger scene is
    referenced as the sibling ``report_topology.html`` instead.
    """
    cap = ('<figcaption>Final design — drag to rotate, scroll to zoom'
           '</figcaption>')
    if scene is not None and scene.is_file():
        try:
            inline = scene.stat().st_size <= MAX_INLINE_SCENE_BYTES
            doc = scene.read_text(encoding="utf-8") if inline else ""
        except OSError:
            inline, doc = False, ""
        if inline and doc:
            attr = f'srcdoc="{escape(doc, quote=True)}"'   # self-contained, offline
        else:
            attr = f'src="{escape(scene.name)}"'           # too big -> sibling file
        return (f'  <figure class="topo"><iframe class="scene" {attr} '
                f'title="Final design (interactive)" loading="lazy"></iframe>'
                f'{cap}</figure>')

    topo_uri = _data_uri(topo) if topo else None
    if topo_uri:
        return (f'  <figure class="topo"><img alt="Final design" src="{topo_uri}">'
                f'<figcaption>Final design</figcaption></figure>')
    return ('  <p class="note">(no rendered image — see the topology files under '
            '<em>Artefacts</em>)</p>')


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _html(s: Summary, work: Path, charts: dict[str, Path],
          scene: Optional[Path], topo: Optional[Path]) -> str:
    now = _dt.datetime.now().isoformat(timespec="seconds")
    rows = "\n".join(
        f"    <tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>"
        for k, v in _rows(s))
    if s.feasible is None:
        badge_cls, badge_txt = "unknown", "FEASIBILITY UNKNOWN"
    elif s.feasible:
        badge_cls, badge_txt = "ok", "FEASIBLE"
    else:
        badge_cls, badge_txt = "bad", "INFEASIBLE"

    parts: list[str] = []
    if s.multi_load:
        parts.append(
            f'  <p class="note">σ_max and displacement are the '
            f'<strong>worst case across {s.n_cases} load cases</strong>; the '
            f'design is feasible only when every case is.</p>')
    if s.stress_excluded:
        parts.append(
            f'  <p class="note">σ_max <strong>excludes {s.stress_excluded} '
            f'element(s)</strong> in the configured stress-exclusion region(s) — '
            f'their von-Mises is ignored for the peak and the feasibility verdict.'
            f'</p>')

    img_blocks: list[str] = []
    for key, title in (("vf", "Volume fraction"),
                       ("sigma", "von-Mises σ_max vs limit"),
                       ("disp", "Displacement vs limit")):
        png = charts.get(key)
        uri = _data_uri(png) if png else None
        if uri:
            img_blocks.append(
                f'    <figure><img alt="{escape(title)}" src="{uri}">'
                f'<figcaption>{escape(title)}</figcaption></figure>')
    charts_html = ("\n".join(img_blocks) if img_blocks
                   else '    <p class="note">(charts unavailable)</p>')

    topo_html = _final_design_html(scene, topo)

    anim_html = _animation_html(work)

    links = "\n".join(
        f'    <li><a href="{escape(name)}">{escape(label)}</a> '
        f'(<code>{escape(name)}</code>)</li>'
        for label, name in _artefacts(work))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>oropt run report</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 2rem auto; max-width: 920px; padding: 0 1rem; color: #1f2328; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
  .meta {{ color: #57606a; font-size: 0.9rem; margin-top: 0; }}
  .badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
           font-weight: 600; font-size: 0.85rem; }}
  .badge.ok {{ background: #dafbe1; color: #1a7f37; }}
  .badge.bad {{ background: #ffebe9; color: #cf222e; }}
  .badge.unknown {{ background: #eaeef2; color: #57606a; }}
  table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
  th, td {{ text-align: left; padding: 0.4rem 0.7rem; border-bottom: 1px solid #d0d7de; }}
  th {{ width: 14rem; color: #57606a; font-weight: 600; }}
  .note {{ color: #57606a; font-size: 0.9rem; }}
  .charts {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
  figure {{ margin: 0; }}
  figure img {{ max-width: 100%; border: 1px solid #d0d7de; border-radius: 6px; }}
  figcaption {{ color: #57606a; font-size: 0.8rem; text-align: center; }}
  .topo img {{ max-width: 620px; }}
  .topo iframe.scene {{ width: 100%; max-width: 620px; height: 460px;
                       border: 1px solid #d0d7de; border-radius: 6px;
                       background: #fff; }}
  .anim img {{ max-width: 620px; }}
  code {{ background: #eaeef2; padding: 0.05rem 0.3rem; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>oropt run report</h1>
  <p class="meta">Optimiser <strong>{escape(s.optimizer)}</strong> &middot;
     generated {escape(now)} &middot;
     <span class="badge {badge_cls}">{badge_txt}</span></p>
{chr(10).join(parts)}
  <h2>Summary</h2>
  <table>
{rows}
  </table>
  <h2>Convergence</h2>
  <div class="charts">
{charts_html}
  </div>
  <h2>Final design</h2>
{topo_html}
  <h2>Evolution</h2>
{anim_html}
  <h2>Artefacts</h2>
  <ul>
{links}
  </ul>
</body>
</html>
"""


def _md(s: Summary, work: Path, charts: dict[str, Path],
        scene: Optional[Path], topo: Optional[Path]) -> str:
    now = _dt.datetime.now().isoformat(timespec="seconds")
    lines = [
        "# oropt run report",
        "",
        f"Optimiser **{s.optimizer}** · generated {now}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {v} |" for k, v in _rows(s)]
    if s.multi_load:
        lines += ["",
                  f"> σ_max and displacement are the **worst case across "
                  f"{s.n_cases} load cases**; the design is feasible only when "
                  f"every case is."]
    if s.stress_excluded:
        lines += ["",
                  f"> σ_max **excludes {s.stress_excluded} element(s)** in the "
                  f"configured stress-exclusion region(s) — their von-Mises is "
                  f"ignored for the peak and the feasibility verdict."]
    lines += ["", "## Convergence", ""]
    any_chart = False
    for key, title in (("vf", "Volume fraction"),
                       ("sigma", "σ_max vs limit"),
                       ("disp", "Displacement vs limit")):
        png = charts.get(key)
        if png is not None:
            lines.append(f"![{title}]({png.name})")
            lines.append("")
            any_chart = True
    if not any_chart:
        lines += ["_(charts unavailable)_", ""]
    lines += ["## Final design", ""]
    if scene is not None and scene.is_file():
        # Markdown can't embed the interactive scene; link the standalone viewer.
        lines += [f"Interactive viewer (rotate / zoom): "
                  f"[`{scene.name}`]({scene.name})", ""]
    elif topo is not None:
        lines += [f"![Final design]({topo.name})", ""]
    else:
        lines += ["_(no rendered image — see the topology files below)_", ""]
    lines += ["## Evolution", ""]
    if (work / ANIM_GIF).is_file():
        lines += [f"![Topology evolution]({ANIM_GIF})", ""]
    else:
        lines += ["_(no evolution animation — see the per-iteration files below)_", ""]
    lines += ["## Artefacts", ""]
    lines += [f"- [{label}]({name})" for label, name in _artefacts(work)]
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def write_report(cfg: Config, work: Path,
                 log: Callable[[str], None] = print) -> Optional[Path]:
    """If enabled, write ``report.html`` and ``report.md`` summarising the run.

    Reads ``status.json`` + ``history.csv`` (+ ``topology_latest.vtu`` for the
    render) from *work*. Returns the path of the HTML report (or the Markdown one
    if only that could be written), else ``None`` (reason logged). Never raises.
    """
    opts = getattr(cfg, "report", None)
    if opts is not None and not getattr(opts, "enabled", True):
        return None
    try:
        work = Path(work)
        status = st.read_status(work)
        history = st.read_history(work)
        if status is None and not history:
            log("[oropt] report: no status.json/history.csv to summarise - skipped")
            return None

        s = _summarise(cfg, status, history)
        timeout = float(getattr(opts, "render_timeout_s", 120.0)
                        if opts is not None else 120.0)
        charts = (_charts(history, s, work, log)
                  if (opts is None or getattr(opts, "charts", True)) else {})
        # Final design: try the interactive viewer first; only spend a second
        # subprocess on the static PNG when it isn't available, so a report always
        # carries *some* final-design visual (the PNG also keeps report.html
        # self-contained when the scene is too big to inline).
        scene = (_export_topology_scene(work, timeout, log)
                 if (opts is None or getattr(opts, "interactive_topology", True))
                 else None)
        want_png = (opts is None or getattr(opts, "render_topology", True))
        topo = (_render_topology(work, timeout, log)
                if (scene is None and want_png) else None)

        written: list[Path] = []
        html_path = work / REPORT_HTML
        try:
            html_path.write_text(_html(s, work, charts, scene, topo),
                                 encoding="utf-8")
            written.append(html_path)
        except OSError as exc:
            log(f"[oropt] report: could not write {REPORT_HTML}: {exc}")
        md_path = work / REPORT_MD
        try:
            md_path.write_text(_md(s, work, charts, scene, topo), encoding="utf-8")
            written.append(md_path)
        except OSError as exc:
            log(f"[oropt] report: could not write {REPORT_MD}: {exc}")
        if not written:
            return None
        log(f"[oropt] report: wrote {', '.join(p.name for p in written)} "
            f"({s.optimizer}, {_pct(s.mass_removed_pct, 1)} mass removed, "
            f"{s.iterations} iters)")
        return written[0]
    except Exception as exc:  # noqa: BLE001  (best-effort: never fail the run)
        log(f"[oropt] report: unexpected error: {exc} - skipped")
        return None

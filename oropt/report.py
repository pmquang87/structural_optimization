"""Best-effort automatic post-run summary report for an oropt run.

After a run finishes, the loop calls :func:`write_report` to write a
human-readable summary of the run into the run folder:

* ``report.html`` — a self-contained page (the hover-interactive convergence charts
  and a render of the final design are embedded inline, so it can be
  e-mailed/archived as one file) with the run summary table, feasibility verdict
  and artefact links;
* ``report.md``   — the same numbers as lightweight Markdown (static charts/render
  referenced as sibling ``report_*.png`` files).

The "final design" is the **last feasible iteration**, not the last one: a gradient
optimiser oscillates across the constraint boundary near convergence, so the last
iteration is often infeasible (peak stress just over the allowable). The report
headlines — and renders (``topology_iterNNNN.vtu``) — that last-feasible design, and
notes where the run actually ended (see :func:`_pick_reported`).

It only *reads* the artefacts the loop already wrote (``status.json``,
``history.csv``, the ``topology_*.vtu`` snapshots), so it never touches the run.
Charts need matplotlib and the topology render needs an off-screen pyvista; both are
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

import argparse
import base64
import datetime as _dt
import json
import math
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Optional

from . import status as st
from ._render import run_render
from .animate import ANIM_GIF
from .config import Config
from .mesh import overlay_primitives

REPORT_HTML = "report.html"
REPORT_MD = "report.md"
TOPOLOGY_PNG = "report_topology.png"
TOPOLOGY_SCENE_HTML = "report_topology.html"   # standalone interactive VTK.js scene
OVERLAY_SPEC_JSON = "report_boxes.json"        # growth-region outlines for the render

# The evolution GIF and the interactive scene are embedded inline so report.html
# stays a single emailable file — but only while each is small enough; above the
# cap we reference the sibling file instead so the HTML doesn't balloon. (The
# VTK.js scene bundles vtk.js, so it is typically a few MB.)
MAX_INLINE_GIF_BYTES = 8 * 1024 * 1024
MAX_INLINE_SCENE_BYTES = 12 * 1024 * 1024

# Optional growth-region overlay for the isolated render subprocesses. When a
# boxes spec (JSON of oropt.mesh.overlay_primitives) is passed as the 3rd argv,
# _overlay adds a red wireframe outline of each region to the plotter — so the
# report shows where material was allowed to grow, like the Monitor's 3D view. No
# 3rd argv (a normal run) -> _overlay is a no-op and the render is byte-identical.
_OVERLAY_SNIPPET = (
    "def _overlay(p, argv):\n"
    "    if len(argv) < 4:\n"
    "        return\n"
    "    import os, json\n"
    "    import numpy as np\n"
    "    if not argv[3] or not os.path.exists(argv[3]):\n"
    "        return\n"
    "    for pr in json.loads(open(argv[3], encoding='utf-8').read()):\n"
    "        k = pr['kind']\n"
    "        if k in ('box', 'polyhedron'):\n"
    "            pts = np.asarray(pr['corners'], dtype=float)\n"
    "            lines = np.hstack([[2, i, j] for i, j in pr['edges']]).astype(int)\n"
    "            m = pv.PolyData(pts, lines=lines)\n"
    "        elif k == 'sphere':\n"
    "            m = pv.Sphere(radius=pr['radius'], center=pr['center'])\n"
    "        else:\n"
    "            a = np.asarray(pr['p1'], dtype=float)\n"
    "            b = np.asarray(pr['p2'], dtype=float)\n"
    "            m = pv.Cylinder(center=(a + b) / 2.0, direction=b - a,\n"
    "                            radius=pr['radius'],\n"
    "                            height=float(np.linalg.norm(b - a)))\n"
    "        p.add_mesh(m, color='red', style='wireframe', line_width=2,\n"
    "                   opacity=0.7)\n"
)
# Run by an isolated subprocess (same interpreter, which already has pyvista) so a
# hard VTK/OpenGL crash on a headless box can never bring the run down — it shows
# up here only as a non-zero exit code. argv -> [vtu_in, png_out, boxes_json?].
_RENDER_RUNNER = (
    "import sys, pyvista as pv\n"
    "pv.OFF_SCREEN = True\n"
    + _OVERLAY_SNIPPET +
    "g = pv.read(sys.argv[1])\n"
    "s = 'sensitivity' if 'sensitivity' in g.cell_data else None\n"
    "p = pv.Plotter(window_size=[900, 600], off_screen=True)\n"
    "p.add_mesh(g, scalars=s, cmap='viridis', show_edges=False)\n"
    "_overlay(p, sys.argv)\n"
    "p.view_isometric(); p.background_color = 'white'\n"
    "p.screenshot(sys.argv[2]); p.close()\n"
)
# Same scene as the PNG render, but exported as an *interactive* standalone VTK.js
# page (zoom/rotate) via pyvista's export_html instead of a screenshot. Also run in
# the isolated subprocess so a GL/driver crash stays contained. argv -> [vtu_in,
# html_out, boxes_json?]. Needs the optional trame export backend (see
# _scene_backend_available).
_SCENE_RUNNER = (
    "import sys, pyvista as pv\n"
    "pv.OFF_SCREEN = True\n"
    + _OVERLAY_SNIPPET +
    "g = pv.read(sys.argv[1])\n"
    "s = 'sensitivity' if 'sensitivity' in g.cell_data else None\n"
    "p = pv.Plotter(window_size=[900, 600], off_screen=True)\n"
    "p.add_mesh(g, scalars=s, cmap='viridis', show_edges=False)\n"
    "_overlay(p, sys.argv)\n"
    "p.view_isometric(); p.background_color = 'white'\n"
    "p.export_html(sys.argv[2]); p.close()\n"
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


def _json_num(x) -> Optional[float]:
    """A JSON-serialisable float, or ``None`` for NaN (JSON has no NaN literal)."""
    v = _f(x)
    return None if _isnan(v) else float(v)


# --------------------------------------------------------------------------- #
# feasible-design selection
# --------------------------------------------------------------------------- #
def _is_feasible_row(row: dict) -> bool:
    """Whether a ``history.csv`` row's ``feasible`` cell is truthy."""
    return str(row.get("feasible", "")).strip().lower() in ("true", "1", "yes")


def _iter_num(row: dict) -> int:
    """The row's integer iteration number (``-1`` when missing/unparseable)."""
    n = _f(row.get("iteration"))
    return int(n) if not _isnan(n) else -1


def _pick_reported(history: list[dict]) -> Optional[dict]:
    """The history row for the design the report should headline: the
    **highest-numbered feasible iteration**, or the last iteration when none is
    feasible (``None`` only when there is no history at all).

    A gradient optimiser oscillates back and forth across the constraint boundary
    near convergence, so the *last* iteration is frequently infeasible (peak stress
    just over the allowable) while the real deliverable is the last iteration that
    satisfied every constraint. Reporting that one keeps the headline design — and
    the rendered topology — feasible; the caller still notes where the run actually
    ended."""
    if not history:
        return None
    feasible = [r for r in history if _is_feasible_row(r)]
    return max(feasible, key=_iter_num) if feasible else history[-1]


def reported_iteration(work: Path) -> int:
    """The iteration number the report headlines/renders — the last feasible one,
    else the last iteration, and ``-1`` when there is no ``history.csv``.

    Reads the run's history from *work* and applies the same selection as
    :func:`_pick_reported`. Public so other post-run steps (e.g. the d3plot
    conversion) can target the **same** design the report shows."""
    row = _pick_reported(st.read_history(work))
    return _iter_num(row) if row is not None else -1


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
    # Absolute design volume V0 (deck units) and material density/cost, so the
    # report can turn the volume fractions into mass. All 0 when unknown (no
    # density configured, or a pre-upgrade status without design_volume).
    design_volume: float = 0.0
    material_density: float = 0.0
    material_cost_per_mass: float = 0.0
    # Provenance of the reported design (see _pick_reported): the design metrics
    # above describe iteration ``reported_iteration`` — the last *feasible* one —
    # while the run actually ended at ``last_iteration``. They differ when the
    # optimiser oscillated across the constraint boundary near convergence.
    reported_iteration: int = -1
    last_iteration: int = -1
    all_infeasible: bool = False     # no iteration was feasible -> reported == last
    last_sigma_max: float = float("nan")
    last_feasible: Optional[bool] = None

    @property
    def multi_load(self) -> bool:
        return self.n_cases > 1

    @property
    def ended_mid_oscillation(self) -> bool:
        """True when the run's last iteration is not the reported (feasible) one."""
        return (self.reported_iteration >= 0
                and self.reported_iteration != self.last_iteration)


def _summarise(cfg: Config, status: Optional[st.Status],
               history: list[dict]) -> Summary:
    """Reduce ``status.json`` + ``history.csv`` to the report's scalar summary.

    The design metrics (volume fraction, σ_max, displacement, feasibility) describe
    the **last feasible iteration** (:func:`_pick_reported`), not the last one — the
    last iteration is frequently infeasible when the optimiser oscillates at the
    constraint boundary. The limits (σ_allow / d_allow are constant across
    iterations) come from ``status.json`` (fallback: the config), and the run's very
    last iteration is recorded too so the report can flag a mid-oscillation ending.
    """
    vfs = [_f(r.get("volume_fraction")) for r in history]
    vfs = [v for v in vfs if not _isnan(v)]
    walls = [_f(r.get("iter_wall_s")) for r in history]
    total_wall = sum(w for w in walls if not _isnan(w))
    start_vf = vfs[0] if vfs else float("nan")

    reported = _pick_reported(history)
    last = history[-1] if history else {}
    all_infeasible = bool(history) and not any(_is_feasible_row(r) for r in history)

    # Design metrics from the reported (last feasible) iteration's history row.
    # With no history at all, status.json — which reflects the last iteration — is
    # the only source, so fall back to it.
    if reported is not None:
        final_vf = _f(reported.get("volume_fraction"))
        sigma_max = _f(reported.get("sigma_max"))
        disp = _f(reported.get("disp"))
        feasible: Optional[bool] = _is_feasible_row(reported)
        reported_iter = _iter_num(reported)
    elif status is not None:
        final_vf = _f(status.volume_fraction)
        sigma_max = _f(status.sigma_max)
        disp = _f(status.disp)
        feasible = bool(status.feasible)
        reported_iter = int(status.iteration)
    else:
        final_vf = sigma_max = disp = float("nan")
        feasible = None
        reported_iter = -1

    if not _isnan(start_vf) and not _isnan(final_vf) and start_vf > 0:
        mass_removed = (start_vf - final_vf) / start_vf * 100.0
    else:
        mass_removed = float("nan")

    # Limits are constant across iterations, so status.json is authoritative; fall
    # back to the primary load case's limits (or NaN) only when it's absent.
    cases = cfg.load_case_list()
    primary = cases[0] if cases else None
    sig_fallback = (primary.sigma_allow if primary and primary.sigma_allow is not None
                    else float("nan"))
    # The primary case's tightest displacement limit (worst-case), or NaN if none.
    d_limits = ([dc.d_allow for dc in primary.disp_constraints
                 if dc.d_allow is not None] if primary else [])
    d_fallback = min(d_limits) if d_limits else float("nan")

    def _limit(attr: str, fallback: float) -> float:
        if status is not None and not _isnan(_f(getattr(status, attr))):
            return float(getattr(status, attr))
        return fallback

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
        sigma_max=sigma_max,
        sigma_allow=_limit("sigma_allow", sig_fallback),
        disp=disp,
        d_allow=_limit("d_allow", d_fallback),
        total_wall_s=total_wall,
        n_cases=len(cfg.load_cases),
        stress_excluded=(int(getattr(status, "stress_excluded_elems", 0))
                         if status is not None else 0),
        design_volume=(_f(getattr(status, "design_volume", 0.0))
                       if status is not None else 0.0),
        material_density=float(getattr(cfg.model, "material_density", 0.0) or 0.0),
        material_cost_per_mass=float(getattr(cfg.model, "material_cost_per_mass", 0.0) or 0.0),
        reported_iteration=reported_iter,
        last_iteration=(_iter_num(last) if last else reported_iter),
        all_infeasible=all_infeasible,
        last_sigma_max=(_f(last.get("sigma_max")) if last else sigma_max),
        last_feasible=(_is_feasible_row(last) if last else feasible),
    )


def _rows(s: Summary) -> list[tuple[str, str]]:
    """(label, value) pairs for the summary table, shared by HTML and Markdown."""
    sigma = f"{_num(s.sigma_max, 1)} / {_num(s.sigma_allow, 1)} MPa"
    disp = f"{_num(s.disp, 4)} / {_num(s.d_allow, 4)} mm"
    if s.feasible is None:
        verdict = "unknown"
    else:
        verdict = "FEASIBLE" if s.feasible else "INFEASIBLE"
    rows = [
        ("Optimizer", s.optimizer),
        ("Termination", f"{s.state or 'n/a'}"
                        + (f" — {s.message}" if s.message else "")),
        ("Iterations", str(s.iterations)),
    ]
    # Which iteration the design metrics below describe. When the run ended
    # mid-oscillation on an infeasible iteration this is an earlier feasible one, so
    # spell it out (and where the run actually ended) rather than silently reporting
    # a different iteration than the run's last.
    if s.reported_iteration >= 0:
        if s.all_infeasible:
            rows.append(("Reported design",
                         f"iteration {s.reported_iteration} "
                         f"(last — no iteration was feasible)"))
        elif s.ended_mid_oscillation:
            rows.append(("Reported design",
                         f"iteration {s.reported_iteration} (last feasible; run "
                         f"ended at iteration {s.last_iteration}, infeasible)"))
        else:
            rows.append(("Reported design", f"iteration {s.reported_iteration}"))
    rows += [
        ("Volume fraction", f"{_num(s.start_vf, 3)} → {_num(s.final_vf, 3)}"),
        # A growth-box run can end with net *added* material (negative removal);
        # flip the label rather than report "-x% removed".
        ("Mass removed" if _isnan(s.mass_removed_pct) or s.mass_removed_pct >= 0
         else "Mass added (net growth)",
         _pct(abs(s.mass_removed_pct), 1)),   # abs(nan) is nan -> still "n/a"
    ]
    # Absolute mass (and cost) when a material density is configured — mass =
    # volume_fraction * V0 * density, in whatever unit system density is given in.
    if s.material_density > 0.0 and s.design_volume > 0.0:
        start_m = s.start_vf * s.design_volume * s.material_density
        final_m = s.final_vf * s.design_volume * s.material_density
        rows.append(("Mass", f"{_num(start_m, 4)} → {_num(final_m, 4)}"))
        if s.material_cost_per_mass > 0.0:
            rows.append(("Cost", f"{_num(start_m * s.material_cost_per_mass, 2)} → "
                                 f"{_num(final_m * s.material_cost_per_mass, 2)}"))
    rows += [
        (f"Final σ_max{' (worst case)' if s.multi_load else ''}", sigma),
        (f"Final disp{' (worst case)' if s.multi_load else ''}", disp),
        ("Feasible", verdict),
        ("Total wall time", _hms(s.total_wall_s)),
    ]
    return rows


def _read_manufacturability(work: Path) -> Optional[dict]:
    """The independent manufacturability audit (oropt.mfg_verify, written to
    ``manufacturability.json`` by the loop when manufacturing constraints are
    active), or ``None`` when absent/unreadable."""
    import json
    p = Path(work) / "manufacturability.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("checks") else None
    except (OSError, ValueError):
        return None


def _manufacturability_rows(work: Path) -> list[tuple[str, str, str]]:
    """(check, result, detail) rows for the manufacturability table, or []."""
    data = _read_manufacturability(work)
    if data is None:
        return []
    out = []
    for c in data.get("checks", []):
        if not c.get("active", True):
            continue
        result = "PASS" if c.get("passed") else "FAIL"
        meas, lim = c.get("measured"), c.get("limit")
        detail = c.get("detail") or (f"measured {meas} vs limit {lim}"
                                     if meas is not None else "")
        out.append((str(c.get("name", "?")), result, str(detail)))
    return out


def _provenance_lines(s: Summary) -> list[str]:
    """Plain-text note(s) about the reported-vs-last iteration (empty when the
    reported design *is* the run's last iteration and it was feasible).

    Surfaces that the headline design is the last feasible iteration while the run
    itself ended on a later, infeasible one (mid-oscillation) — or that no iteration
    was feasible at all, so the fallback last-iteration design is over the limit."""
    if s.all_infeasible:
        return [f"No iteration satisfied every constraint — the reported design is "
                f"the last iteration ({s.last_iteration}) and is INFEASIBLE "
                f"(σ_max {_num(s.sigma_max, 1)} MPa vs the {_num(s.sigma_allow, 1)} "
                f"MPa limit)."]
    if s.ended_mid_oscillation:
        return [f"The run ended mid-oscillation at iteration {s.last_iteration} "
                f"(INFEASIBLE, σ_max {_num(s.last_sigma_max, 1)} MPa). The reported "
                f"final design is the last feasible iteration "
                f"({s.reported_iteration})."]
    return []


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
# interactive convergence charts (self-contained inline SVG, HTML only)
# --------------------------------------------------------------------------- #
# Hover-interactive charts for report.html, mirroring the GUI Monitor's three
# per-iteration line charts (volume fraction; σ_max vs limit; displacement vs
# limit). Rather than inline a multi-MB charting library (Plotly/Chart.js), the
# series are drawn as a compact self-contained SVG by ~120 lines of dependency-free
# vanilla JS injected below — so report.html stays a single offline file (no CDN,
# no network) exactly like its inlined images/GIF/VTK.js scene, and the hover
# tooltip can show precisely the iteration number + value the Monitor exposes. The
# data is injected as a plain-Python JSON literal at ``__OROPT_CHART_DATA__``
# (numpy types must never leak in — cf. the numpy-intc→json.dumps GIF regression).
_INTERACTIVE_CHART_JS = r"""
(function () {
  "use strict";
  var DATA = __OROPT_CHART_DATA__;
  var NS = "http://www.w3.org/2000/svg";
  var C = { line: "#1f6feb", limit: "#cf222e", grid: "#e1e6eb",
            axis: "#8c959f", text: "#57606a", reported: "#1a7f37" };
  function el(name, attrs) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) { e.setAttribute(k, attrs[k]); }
    return e;
  }
  function fmt(v, d) {
    return (v === null || v === undefined || isNaN(v)) ? "n/a" : Number(v).toFixed(d);
  }
  function ticks(lo, hi, n) {
    var span = hi - lo;
    if (!(span > 0)) { return [lo]; }
    var step = Math.pow(10, Math.floor(Math.log(span / n) / Math.LN10));
    var err = (span / n) / step;
    if (err >= 7.5) { step *= 10; } else if (err >= 3) { step *= 5; }
    else if (err >= 1.5) { step *= 2; }
    var out = [], t = Math.ceil(lo / step) * step;
    for (; t <= hi + step * 1e-6; t += step) { out.push(Math.round(t / step) * step); }
    return out;
  }
  function build(box, spec, iters, reported) {
    var W = 480, H = 300, m = { l: 60, r: 16, t: 14, b: 40 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b, vals = spec.values, dec = spec.decimals;
    var ys = [], i;
    for (i = 0; i < vals.length; i++) { if (vals[i] !== null) { ys.push(vals[i]); } }
    if (spec.limit !== null) { ys.push(spec.limit); }
    if (!ys.length) { box.innerHTML = '<p class="note">(no data)</p>'; return; }
    var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    if (ymin === ymax) { ymin -= Math.abs(ymin) * 0.1 + 1; ymax += Math.abs(ymax) * 0.1 + 1; }
    var pad = (ymax - ymin) * 0.08; ymin -= pad; ymax += pad;
    var xmin = Math.min.apply(null, iters), xmax = Math.max.apply(null, iters);
    if (xmin === xmax) { xmin -= 0.5; xmax += 0.5; }
    function X(v) { return m.l + (v - xmin) / (xmax - xmin) * iw; }
    function Y(v) { return m.t + (1 - (v - ymin) / (ymax - ymin)) * ih; }
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%",
                          "font-family": "system-ui, -apple-system, sans-serif" });
    var yt = ticks(ymin, ymax, 5), k, e;
    for (k = 0; k < yt.length; k++) {
      var yy = Y(yt[k]); if (yy < m.t - 0.5 || yy > m.t + ih + 0.5) { continue; }
      svg.appendChild(el("line", { x1: m.l, y1: yy, x2: m.l + iw, y2: yy,
                                   stroke: C.grid, "stroke-width": 1 }));
      e = el("text", { x: m.l - 6, y: yy + 3, "text-anchor": "end",
                       "font-size": 10, fill: C.text });
      e.textContent = fmt(yt[k], dec); svg.appendChild(e);
    }
    var xt = ticks(xmin, xmax, 8);
    for (k = 0; k < xt.length; k++) {
      if (xt[k] !== Math.round(xt[k])) { continue; }
      var xx = X(xt[k]); if (xx < m.l - 0.5 || xx > m.l + iw + 0.5) { continue; }
      e = el("text", { x: xx, y: m.t + ih + 16, "text-anchor": "middle",
                       "font-size": 10, fill: C.text });
      e.textContent = String(Math.round(xt[k])); svg.appendChild(e);
    }
    e = el("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle",
                     "font-size": 10, fill: C.text });
    e.textContent = "iteration"; svg.appendChild(e);
    if (spec.limit !== null) {
      var ly = Y(spec.limit);
      svg.appendChild(el("line", { x1: m.l, y1: ly, x2: m.l + iw, y2: ly, stroke: C.limit,
                                   "stroke-width": 1.4, "stroke-dasharray": "5 4" }));
      e = el("text", { x: m.l + iw - 2, y: ly - 4, "text-anchor": "end",
                       "font-size": 10, fill: C.limit });
      e.textContent = spec.limitLabel + " " + fmt(spec.limit, dec); svg.appendChild(e);
    }
    var d = "", started = false;
    for (i = 0; i < vals.length; i++) {
      if (vals[i] === null) { started = false; continue; }
      d += (started ? " L" : "M") + X(iters[i]) + " " + Y(vals[i]); started = true;
    }
    svg.appendChild(el("path", { d: d, fill: "none", stroke: C.line, "stroke-width": 1.6 }));
    for (i = 0; i < vals.length; i++) {
      if (vals[i] === null) { continue; }
      var rep = (iters[i] === reported);
      svg.appendChild(el("circle", { cx: X(iters[i]), cy: Y(vals[i]), r: rep ? 4 : 2.4,
        fill: rep ? C.reported : C.line, stroke: "#fff", "stroke-width": rep ? 1.5 : 0 }));
    }
    var vline = el("line", { x1: 0, y1: m.t, x2: 0, y2: m.t + ih, stroke: C.axis,
                             "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0 });
    var focus = el("circle", { r: 4, fill: "none", stroke: C.line, "stroke-width": 2, opacity: 0 });
    svg.appendChild(vline); svg.appendChild(focus);
    var over = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
    svg.appendChild(over); box.appendChild(svg);
    var tip = document.createElement("div"); tip.className = "ichart-tip"; box.appendChild(tip);
    function nearest(px) {
      var best = -1, bd = 1e18;
      for (var j = 0; j < iters.length; j++) {
        if (vals[j] === null) { continue; }
        var dx = Math.abs(X(iters[j]) - px); if (dx < bd) { bd = dx; best = j; }
      }
      return best;
    }
    over.addEventListener("mousemove", function (ev) {
      var r = svg.getBoundingClientRect();
      var j = nearest((ev.clientX - r.left) / r.width * W); if (j < 0) { return; }
      var cx = X(iters[j]), cy = Y(vals[j]);
      vline.setAttribute("x1", cx); vline.setAttribute("x2", cx); vline.setAttribute("opacity", 1);
      focus.setAttribute("cx", cx); focus.setAttribute("cy", cy); focus.setAttribute("opacity", 1);
      var html = "iter " + iters[j] + "<br><b>" + fmt(vals[j], dec) + "</b>" +
                 (spec.unit ? " " + spec.unit : "");
      if (spec.limit !== null) {
        var bad = vals[j] > spec.limit;
        html += '<br><span style="color:' + (bad ? C.limit : C.reported) + '">' +
                (bad ? "over " : "within ") + spec.limitLabel + "</span>";
      }
      tip.innerHTML = html; tip.style.opacity = 1;
      tip.style.left = (cx / W * r.width) + "px";
      tip.style.top = (cy / H * r.height) + "px";
    });
    over.addEventListener("mouseleave", function () {
      vline.setAttribute("opacity", 0); focus.setAttribute("opacity", 0); tip.style.opacity = 0;
    });
  }
  function boot() {
    for (var i = 0; i < DATA.series.length; i++) {
      var spec = DATA.series[i], box = document.getElementById("ichart-" + spec.key);
      if (box) { build(box, spec, DATA.iter, DATA.reported); }
    }
  }
  if (document.readyState !== "loading") { boot(); }
  else { document.addEventListener("DOMContentLoaded", boot); }
})();
"""

_ICHART_TITLES = {
    "vf": "Volume fraction",
    "sigma": "von-Mises σ_max vs limit",
    "disp": "Displacement vs limit",
}


def _chart_payload(history: list[dict], s: Summary) -> str:
    """The JSON literal the inline chart JS reads (all plain-Python; NaN -> null).

    ``<`` is escaped so a value can never break out of the ``<script>`` tag."""
    iters = [_iter_num(r) for r in history]

    def col(key: str) -> list[Optional[float]]:
        return [_json_num(r.get(key)) for r in history]

    payload = {
        "iter": iters,
        "reported": int(s.reported_iteration),
        "series": [
            {"key": "vf", "values": col("volume_fraction"), "decimals": 3,
             "unit": "", "limit": None, "limitLabel": None},
            {"key": "sigma", "values": col("sigma_max"), "decimals": 1,
             "unit": "MPa", "limit": _json_num(s.sigma_allow), "limitLabel": "σ_allow"},
            {"key": "disp", "values": col("disp"), "decimals": 4,
             "unit": "mm", "limit": _json_num(s.d_allow), "limitLabel": "d_allow"},
        ],
    }
    return json.dumps(payload).replace("<", "\\u003c")


def _interactive_charts_html(history: list[dict], s: Summary,
                             png_charts: dict[str, Path]) -> str:
    """The Convergence section's inner HTML: three hover-interactive inline-SVG
    charts (built by :data:`_INTERACTIVE_CHART_JS`), each with the static PNG as a
    ``<noscript>`` fallback so the report still shows charts with JS disabled."""
    if not history:
        return '    <p class="note">(charts unavailable)</p>'
    figs: list[str] = []
    for key in ("vf", "sigma", "disp"):
        title = _ICHART_TITLES[key]
        uri = _data_uri(png_charts[key]) if key in png_charts else None
        noscript = (f'<noscript><img alt="{escape(title)}" src="{uri}"></noscript>'
                    if uri else "")
        figs.append(
            f'    <figure class="ichart-fig"><div class="ichart" '
            f'id="ichart-{key}"></div>{noscript}'
            f'<figcaption>{escape(title)}</figcaption></figure>')
    script = ("  <script>" + _INTERACTIVE_CHART_JS.replace(
        "__OROPT_CHART_DATA__", _chart_payload(history, s)) + "</script>")
    return "\n".join(figs) + "\n" + script


# --------------------------------------------------------------------------- #
# final-topology render (off-screen pyvista, best-effort)
# --------------------------------------------------------------------------- #
def _write_overlay_spec(work: Path, cfg: Config,
                        log: Callable[[str], None]) -> Optional[Path]:
    """Write the growth-region outline spec (``report_boxes.json``) for the render.

    Serialises :func:`oropt.mesh.overlay_primitives` of the config's growth regions
    so the isolated render subprocess can overlay them. Returns the path, or
    ``None`` when there are no drawable regions (so a normal run passes no overlay
    and its render stays byte-identical). Best-effort: never raises."""
    try:
        boxes = getattr(getattr(cfg, "model", None), "growth_boxes", None) or []
        prims = overlay_primitives(boxes)
        if not prims:
            return None
        dest = Path(work) / OVERLAY_SPEC_JSON
        dest.write_text(json.dumps(prims), encoding="utf-8")
        return dest
    except Exception as exc:  # noqa: BLE001
        log(f"[oropt] report: could not write growth-region overlay: {exc}")
        return None


def _reported_topology_src(work: Path, reported_iteration: int) -> Path:
    """The VTU to render as the 'final design': the reported iteration's immutable
    per-iteration snapshot when it exists, else ``topology_latest.vtu``.

    Per-iteration snapshots are written to the run root as ``topology_iterNNNN.vtu``
    (:func:`oropt.status.topology_snapshot_name`); a nested ``iter_NNNN/`` copy is
    checked too for robustness. ``topology_latest.vtu`` is the *last* iteration,
    which may be infeasible, so it is only the fallback when the chosen snapshot is
    missing (e.g. an older run that predates per-iteration snapshots)."""
    work = Path(work)
    if reported_iteration >= 0:
        name = st.topology_snapshot_name(reported_iteration)
        for cand in (work / name,
                     work / f"iter_{reported_iteration:04d}" / name):
            if cand.is_file():
                return cand
    return work / st.TOPOLOGY


def _render_topology(work: Path, timeout_s: float,
                     log: Callable[[str], None],
                     boxes_spec: Optional[Path] = None,
                     src: Optional[Path] = None) -> Optional[Path]:
    """Off-screen render of the final design to ``report_topology.png``.

    Renders *src* (the reported feasible iteration's topology snapshot; defaults to
    ``topology_latest.vtu``) exactly like the GUI's off-screen pyvista view, but in
    an *isolated subprocess* so a hard VTK/OpenGL crash on a headless machine (no GL
    context) can never abort the run — it just shows up as a non-zero exit code and
    we fall back to linking the topology files. When *boxes_spec* is given, each
    growth region is overlaid as a red wireframe outline. Returns the PNG path, or
    ``None`` (reason logged) when there is nothing to render or the render
    subprocess fails/times out/crashes.
    """
    src = Path(src) if src is not None else Path(work) / st.TOPOLOGY
    if not src.is_file():
        log(f"[oropt] report: no {src.name} to render - topology image skipped")
        return None
    dest = Path(work) / TOPOLOGY_PNG
    dest.unlink(missing_ok=True)        # don't mistake a stale PNG for a fresh one
    args = [src, dest] if boxes_spec is None else [src, dest, boxes_spec]
    result = run_render(_RENDER_RUNNER, args, timeout_s)
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
    the offline HTML export — ``trame_vtk`` (the vtk.js exporter) and a
    nest-asyncio module (to launch trame's server synchronously) — predicts
    whether the export can work. Both ship with the optional ``report3d`` extra
    (``pyvista[jupyter]``); when either is missing we skip a doomed subprocess and
    fall straight back to the static PNG. pyvista >= 0.47 depends on
    ``nest_asyncio2`` but 0.43–0.46 (inside our declared support range) use the
    original ``nest_asyncio`` — accept either, or the viewer is silently dead on
    a correctly-installed older stack.
    """
    import importlib.util
    if importlib.util.find_spec("trame_vtk") is None:
        return False
    return any(importlib.util.find_spec(m) is not None
               for m in ("nest_asyncio2", "nest_asyncio"))


def _export_topology_scene(work: Path, timeout_s: float,
                           log: Callable[[str], None],
                           boxes_spec: Optional[Path] = None,
                           src: Optional[Path] = None) -> Optional[Path]:
    """Export an interactive (zoom/rotate) VTK.js scene to ``report_topology.html``.

    Renders the same *src* scene as :func:`_render_topology` (the reported feasible
    iteration's snapshot; defaults to ``topology_latest.vtu``), but via pyvista's
    ``export_html`` into a standalone, offline interactive viewer — so the report's
    final design can be orbited/zoomed like the GUI's Monitor tab. Runs in the
    *isolated subprocess* (crash containment). When *boxes_spec* is given, each
    growth region is overlaid as a red wireframe outline. Returns the scene path, or
    ``None`` (reason logged) when the optional trame backend is missing, there is
    nothing to render, or the export fails/times out/crashes — the caller then falls
    back to the static PNG. Never raises.
    """
    if not _scene_backend_available():
        log("[oropt] report: interactive 3D viewer needs the optional 'report3d' "
            "extra (trame-vtk) - using a static image "
            "(pip install \"oropt[report3d]\")")
        return None
    src = Path(src) if src is not None else Path(work) / st.TOPOLOGY
    if not src.is_file():
        log(f"[oropt] report: no {src.name} to render - interactive view skipped")
        return None
    dest = Path(work) / TOPOLOGY_SCENE_HTML
    dest.unlink(missing_ok=True)        # don't mistake a stale scene for a fresh one
    args = [src, dest] if boxes_spec is None else [src, dest, boxes_spec]
    result = run_render(_SCENE_RUNNER, args, timeout_s)
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


def _design_caption(s: Summary) -> str:
    """The final-design caption, naming which iteration is rendered.

    The rendered design is the reported (last feasible) iteration — see
    :func:`_pick_reported` — so spell out that iteration number, and whether it was
    feasible, right on the 3D view (a ``-1`` iteration, i.e. no history, drops the
    suffix)."""
    if s.reported_iteration < 0:
        return "Final design"
    if s.all_infeasible:
        return (f"Final design — iteration {s.reported_iteration} "
                f"(last iteration, infeasible)")
    return f"Final design — iteration {s.reported_iteration} (last feasible)"


def _final_design_html(scene: Optional[Path], topo: Optional[Path],
                       label: str) -> str:
    """The 'Final design' block: interactive viewer if exported, else the static
    PNG, else a note pointing at the topology files.

    *label* names the rendered iteration (see :func:`_design_caption`) and is used
    as the figure caption. The interactive scene (zoom/rotate, like the Monitor tab)
    is inlined via an ``<iframe srcdoc>`` while it's under
    :data:`MAX_INLINE_SCENE_BYTES` so report.html stays one self-contained,
    offline-viewable file; a larger scene is referenced as the sibling
    ``report_topology.html`` instead.
    """
    label = escape(label)
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
                f'<figcaption>{label} — drag to rotate, scroll to zoom'
                f'</figcaption></figure>')

    topo_uri = _data_uri(topo) if topo else None
    if topo_uri:
        return (f'  <figure class="topo"><img alt="Final design" src="{topo_uri}">'
                f'<figcaption>{label}</figcaption></figure>')
    return ('  <p class="note">(no rendered image — see the topology files under '
            '<em>Artefacts</em>)</p>')


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _html(s: Summary, work: Path, charts: dict[str, Path],
          scene: Optional[Path], topo: Optional[Path],
          history: list[dict]) -> str:
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
    for line in _provenance_lines(s):
        parts.append(f'  <p class="note">{escape(line)}</p>')
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
    mfg_rows = _manufacturability_rows(work)
    if mfg_rows:
        overall = "PASS" if all(r == "PASS" for _, r, _ in mfg_rows) else "FAIL"
        trs = "\n".join(
            f'    <tr><td>{escape(name)}</td><td>{res}</td>'
            f'<td>{escape(detail)}</td></tr>'
            for name, res, detail in mfg_rows)
        parts.append(
            '  <h2>Manufacturability audit</h2>\n'
            f'  <p class="note">Independent geometric re-check of the final design '
            f'against the configured manufacturing constraints — '
            f'<strong>{overall}</strong>.</p>\n'
            '  <table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>\n'
            f'{trs}\n  </table>')

    # Hover-interactive inline-SVG charts (iteration + value on hover), like the
    # Monitor tab; the static PNGs remain as the <noscript> fallback.
    charts_html = _interactive_charts_html(history, s, charts)

    topo_html = _final_design_html(scene, topo, _design_caption(s))

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
  .charts {{ display: flex; flex-wrap: wrap; gap: 0.75rem; }}
  figure {{ margin: 0; }}
  figure img {{ max-width: 100%; border: 1px solid #d0d7de; border-radius: 6px; }}
  figcaption {{ color: #57606a; font-size: 0.8rem; text-align: center; }}
  .ichart-fig {{ flex: 1 1 300px; max-width: 480px; }}
  .ichart {{ position: relative; width: 100%; }}
  .ichart svg {{ display: block; width: 100%; height: auto;
                border: 1px solid #d0d7de; border-radius: 6px; background: #fff; }}
  .ichart noscript img {{ max-width: 100%; }}
  .ichart-tip {{ position: absolute; pointer-events: none; z-index: 2; opacity: 0;
                transform: translate(-50%, -135%); white-space: nowrap;
                background: #1f2328; color: #fff; font-size: 11px; line-height: 1.3;
                padding: 4px 7px; border-radius: 5px; transition: opacity 0.05s; }}
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
  <p class="meta">Optimizer <strong>{escape(s.optimizer)}</strong> &middot;
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
        f"Optimizer **{s.optimizer}** · generated {now}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {v} |" for k, v in _rows(s)]
    for line in _provenance_lines(s):
        lines += ["", f"> {line}"]
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
    mfg_rows = _manufacturability_rows(work)
    if mfg_rows:
        overall = "PASS" if all(r == "PASS" for _, r, _ in mfg_rows) else "FAIL"
        lines += ["", "## Manufacturability audit", "",
                  f"Independent geometric re-check of the final design against the "
                  f"configured manufacturing constraints — **{overall}**.", "",
                  "| Check | Result | Detail |", "| --- | --- | --- |"]
        lines += [f"| {name} | {res} | {detail} |" for name, res, detail in mfg_rows]
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

    Reads ``status.json`` + ``history.csv`` (+ the last-feasible iteration's
    ``topology_iterNNNN.vtu`` snapshot for the render) from *work*. Returns the path
    of the HTML report (or the Markdown one if only that could be written), else
    ``None`` (reason logged). Never raises.
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
        # Growth-region outlines to overlay on the render (None for a normal run,
        # so its render stays byte-identical). Shared by the interactive scene and
        # the static PNG.
        boxes_spec = _write_overlay_spec(work, cfg, log)
        # Render the reported (last feasible) iteration's snapshot, not
        # topology_latest.vtu (the last iteration, which may be infeasible).
        render_src = _reported_topology_src(work, s.reported_iteration)
        # Final design: try the interactive viewer first; only spend a second
        # subprocess on the static PNG when it isn't available, so a report always
        # carries *some* final-design visual (the PNG also keeps report.html
        # self-contained when the scene is too big to inline).
        scene = (_export_topology_scene(work, timeout, log, boxes_spec, render_src)
                 if (opts is None or getattr(opts, "interactive_topology", True))
                 else None)
        want_png = (opts is None or getattr(opts, "render_topology", True))
        topo = (_render_topology(work, timeout, log, boxes_spec, render_src)
                if (scene is None and want_png) else None)

        written: list[Path] = []
        html_path = work / REPORT_HTML
        try:
            html_path.write_text(_html(s, work, charts, scene, topo, history),
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


def main(argv=None) -> int:
    """Standalone: ``python -m oropt.report <run_dir>`` to (re)generate the report.

    Headless twin of the GUI's "Re-generate report" button: refresh
    ``report.html`` / ``report.md`` for any existing run folder without
    re-running the optimisation (e.g. after an oropt update improved the
    report). Reads only the artefacts the run already wrote. Prefers the run's
    own frozen ``config_used.yaml`` — the config it *actually* ran with — so the
    summarised optimiser/limits match the run; ``--config`` overrides it.
    """
    ap = argparse.ArgumentParser(
        prog="oropt-report",
        description="(Re)generate report.html / report.md for a finished run "
                    "folder from its status.json / history.csv, without "
                    "re-running the optimisation.")
    ap.add_argument("run_dir", help="run folder containing the run's "
                                    "status.json / history.csv")
    ap.add_argument("--config", default=None,
                    help="YAML config to summarise with (default: the run's "
                         "frozen config_used.yaml, else oropt defaults)")
    ap.add_argument("--render-timeout", type=float, default=None,
                    help="per-render subprocess timeout [s] for the final-design "
                         "view (default: the config's report.render_timeout_s)")
    ap.add_argument("--no-charts", action="store_true",
                    help="skip the convergence charts")
    ap.add_argument("--no-render", action="store_true",
                    help="skip the final-design view (interactive and static)")
    args = ap.parse_args(argv)

    def log(s: str) -> None:
        print(s, flush=True)

    work = Path(args.run_dir)
    if not work.is_dir():
        log(f"[oropt] report: run folder not found: {work}")
        return 1

    cfg: Optional[Config] = None
    src = args.config if args.config else work / "config_used.yaml"
    if args.config or Path(src).is_file():
        try:
            cfg = Config.from_yaml(str(src))
            log(f"[oropt] report: summarising with {src}")
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] report: could not read {src}: {exc}")
            if args.config:      # an explicit config that doesn't load is fatal;
                return 1         # the frozen one just falls back to defaults
    if cfg is None:
        log("[oropt] report: no config found - summarising with oropt defaults "
            "(optimiser/limit rows may not match the run)")
        cfg = Config()

    # The user explicitly asked for a report, so a config that disabled the
    # automatic post-run one must not veto this run of the tool.
    cfg.report.enabled = True
    if args.render_timeout is not None:
        cfg.report.render_timeout_s = args.render_timeout
    if args.no_charts:
        cfg.report.charts = False
    if args.no_render:
        cfg.report.interactive_topology = False
        cfg.report.render_topology = False

    out = write_report(cfg, work, log=log)
    if out is None:
        log("[oropt] report: no report produced (see messages above)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

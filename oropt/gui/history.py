"""Run-history browsing and multi-run comparison (Streamlit-free, hermetic).

The GUI's Monitor / Re-postprocessing tabs used to require *typing* a run-folder
path; nothing enumerated past runs, and a sweep's convergence traces could not be
overlaid. This module owns that logic so the dashboard's browse/compare/download
widgets stay thin render calls:

* :func:`scan_runs` — enumerate every run folder (any directory holding a
  ``status.json``) under a root, newest first, as :class:`RunEntry` headlines;
* :func:`load_history_series` / :func:`overlay_series` — a run's ``history.csv``
  as aligned per-iteration lists, and several runs' traces shaped for a
  ``st.line_chart`` overlay;
* :func:`compare_table` — one headline row per run for a side-by-side table;
* :func:`downloadable_artifacts` — the shippable end products a run folder
  actually holds, for download buttons.

Kept Streamlit-free (like :mod:`oropt.gui.cases` / :mod:`oropt.gui.runstate`) so
it is fast to import and unit-testable without booting the dashboard. Everything
here is best-effort: a malformed / unreadable run is skipped, never raised into
the GUI.
"""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from oropt import status as st_io
from oropt.animate import ANIM_GIF
from oropt.report import REPORT_HTML
from oropt.smoothing import SMOOTHED_BASE

CONFIG_USED = "config_used.yaml"     # the frozen config the loop snapshots per run
MAX_SCAN_DEPTH = 3                   # directory levels below the root to search

# Sidebar-style badge per run state (mirrors the Monitor's vocabulary).
STATE_BADGE = {"idle": "⚪", "running": "🟢", "converged": "✅",
               "failed": "❌", "stopped": "⏸"}

# The aligned per-iteration series load_history_series always returns.
SERIES_KEYS = ("iteration", "volume_fraction", "sigma_max", "disp", "feasible")

# (label, filename) of every shippable end product a run folder may hold, in
# display order. Filenames come from the writers' own constants so they can't
# drift: oropt.report.REPORT_HTML, oropt.smoothing.SMOOTHED_BASE (.stl/.vtp are
# its two output_format extensions), oropt.animate.ANIM_GIF, oropt.status.HISTORY
# and the loop's manufacturability.json (oropt.loop.write_manufacturability).
ARTIFACTS = (
    ("📝 Report", REPORT_HTML),
    ("🧊 Smoothed surface (STL)", f"{SMOOTHED_BASE}.stl"),
    ("🧊 Smoothed surface (VTP)", f"{SMOOTHED_BASE}.vtp"),
    ("🎬 Evolution GIF", ANIM_GIF),
    ("📈 Iteration history (CSV)", st_io.HISTORY),
    ("🏭 Manufacturability audit", "manufacturability.json"),
)

_MIME = {".html": "text/html", ".gif": "image/gif", ".csv": "text/csv",
         ".json": "application/json", ".stl": "model/stl",
         ".vtp": "application/xml"}


@dataclass(frozen=True)
class RunEntry:
    """One past run's headline state, as scanned from its folder."""
    path: Path            # the run folder itself
    name: str             # path relative to the scanned root ("." = the root)
    state: str            # idle | running | converged | failed | stopped
    iteration: int
    max_iter: int
    volume_fraction: float
    sigma_max: float      # NaN before the first iteration
    feasible: bool
    optimizer: str        # from config_used.yaml; "" when unknown
    updated: str          # ISO timestamp (status.json field, or its file mtime)


def _read_optimizer(run_dir: Path) -> str:
    """The run's optimiser from its frozen ``config_used.yaml`` ("" if unknown)."""
    try:
        data = yaml.safe_load((run_dir / CONFIG_USED).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - absent/unreadable/bad YAML -> unknown
        return ""
    if isinstance(data, dict):
        return str(data.get("optimizer") or "").strip().lower()
    return ""


def _run_entry(run_dir: Path, root: Path) -> Optional[RunEntry]:
    """Build a :class:`RunEntry` for *run_dir*, or ``None`` when malformed."""
    try:
        status = st_io.read_status(run_dir)
        if status is None:                      # missing / unparseable status.json
            return None
        updated = str(status.updated or "").strip()
        if not updated:                          # pre-upgrade status: use file mtime
            mtime = (run_dir / st_io.STATUS).stat().st_mtime
            updated = _dt.datetime.fromtimestamp(mtime).isoformat(
                timespec="seconds")
        return RunEntry(
            path=run_dir,
            name=run_dir.relative_to(root).as_posix(),
            state=str(status.state),
            iteration=int(status.iteration),
            max_iter=int(status.max_iter),
            volume_fraction=float(status.volume_fraction),
            sigma_max=float(status.sigma_max),
            feasible=bool(status.feasible),
            optimizer=_read_optimizer(run_dir),
            updated=updated)
    except Exception:  # noqa: BLE001 - a malformed run is skipped, never raised
        return None


def scan_runs(root: Path, limit: int = 50,
              max_depth: int = MAX_SCAN_DEPTH) -> list[RunEntry]:
    """Every run folder under *root* (any directory holding a ``status.json``),
    newest first, at most *limit* entries.

    The walk is bounded to *max_depth* directory levels below the root (the root
    itself counts as a run folder too) and skips hidden directories. Malformed or
    unreadable entries are dropped; a missing/unreadable root reads as no runs.
    Sorted by the ``updated`` ISO timestamp, descending — lexicographic order is
    chronological for ISO-8601 strings.
    """
    root = Path(root)
    try:
        if not root.is_dir():
            return []
    except OSError:
        return []
    entries: list[RunEntry] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _e: None):
        d = Path(dirpath)
        depth = len(d.relative_to(root).parts)
        if depth >= max_depth:
            dirnames[:] = []                     # bound the walk
        else:
            dirnames[:] = sorted(n for n in dirnames if not n.startswith("."))
        if st_io.STATUS in filenames:
            e = _run_entry(d, root)
            if e is not None:
                entries.append(e)
    entries.sort(key=lambda e: e.updated, reverse=True)
    return entries[:limit]


def _float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def load_history_series(run_dir: Path) -> dict[str, list]:
    """A run's ``history.csv`` as aligned per-iteration lists.

    Returns the :data:`SERIES_KEYS` columns (``iteration`` as int, the metrics as
    float — ``NaN`` for an unparseable cell so the lists stay aligned — and
    ``feasible`` as bool). Rows without a parseable iteration are dropped;
    a missing/unreadable file yields all-empty lists.
    """
    out: dict[str, list] = {k: [] for k in SERIES_KEYS}
    try:
        rows = st_io.read_history(run_dir)
    except Exception:  # noqa: BLE001 - unreadable history reads as empty
        return out
    for row in rows:
        try:
            it = int(float(row["iteration"]))
        except (KeyError, TypeError, ValueError):
            continue                             # header glitch / partial row
        out["iteration"].append(it)
        out["volume_fraction"].append(_float(row.get("volume_fraction")))
        out["sigma_max"].append(_float(row.get("sigma_max")))
        out["disp"].append(_float(row.get("disp")))
        out["feasible"].append(_bool(row.get("feasible")))
    return out


def overlay_series(named_series: dict[str, dict[str, list]],
                   metric: str) -> dict[str, dict[int, float]]:
    """Several runs' traces of one *metric*, shaped for a line-chart overlay.

    ``{run name: {iteration: value}}`` — exactly what ``pandas.DataFrame``
    accepts to build one column per run indexed by iteration (runs of different
    lengths simply leave the other columns' tail as NaN). NaN samples and runs
    with no data for the metric are dropped.
    """
    out: dict[str, dict[int, float]] = {}
    for name, series in named_series.items():
        pts = {i: v for i, v in zip(series.get("iteration", []),
                                    series.get(metric, []))
               if v == v}                        # drop NaN samples
        if pts:
            out[name] = pts
    return out


def compare_table(entries: list[RunEntry]) -> list[dict]:
    """One headline row per run for the side-by-side compare table."""
    return [{
        "run": e.name,
        "state": f"{STATE_BADGE.get(e.state, '•')} {e.state}",
        "optimizer": e.optimizer or "—",
        "iteration": f"{e.iteration}/{e.max_iter}",
        "volume_fraction": e.volume_fraction,
        "σ_max [MPa]": e.sigma_max,
        "feasible": "✅" if e.feasible else "⚠️",
        "updated": e.updated,
    } for e in entries]


def entry_label(e: RunEntry) -> str:
    """One-line badge + headline scalars for a selectbox / browser row."""
    sig = "—" if e.sigma_max != e.sigma_max else f"{e.sigma_max:.1f}"
    label = (f"{STATE_BADGE.get(e.state, '•')} {e.name} · "
             f"it {e.iteration}/{e.max_iter} · vf {e.volume_fraction:.3f} · "
             f"σ {sig} MPa · {'✅' if e.feasible else '⚠️'}")
    if e.optimizer:
        label += f" · {e.optimizer}"
    return label


def downloadable_artifacts(run_dir: Path) -> list[tuple[str, Path]]:
    """The ``(label, path)`` of every shippable artefact *run_dir* actually holds
    (:data:`ARTIFACTS` order); absent files are simply not listed."""
    run_dir = Path(run_dir)
    out: list[tuple[str, Path]] = []
    for label, fname in ARTIFACTS:
        p = run_dir / fname
        try:
            if p.is_file():
                out.append((label, p))
        except OSError:
            continue
    return out


def artifact_mime(path: str | Path) -> str:
    """The download MIME type for an artefact (octet-stream when unknown)."""
    return _MIME.get(Path(path).suffix.lower(), "application/octet-stream")

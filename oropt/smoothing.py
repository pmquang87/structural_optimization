"""Best-effort surface smoothing of the optimised geometry.

After a run finishes, the surface of the final design (the latest
``topology_latest.vtu``) is extracted, smoothed and written as
``topology_smoothed.<ext>`` in the run folder — a clean deliverable for
CAD / 3D-print / review. With :func:`smooth_all_iterations` the same is done for
*every* per-iteration snapshot (``topology_iterNNNN.vtu`` ->
``topology_smoothed_iterNNNN.<ext>``) so the smoothed shape evolution can be
reviewed too. Uses pyvista, which the loop already requires for its per-iteration
topology output. Never raises: failures are logged and return ``None``/``[]`` so
smoothing can never abort or fail an optimisation run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from . import status as st
from .config import Config

SMOOTHED_BASE = "topology_smoothed"


def _formats(output_format: str) -> list[str]:
    fmt = str(output_format).lower()
    if fmt == "both":
        return ["stl", "vtp"]
    return [fmt] if fmt in ("stl", "vtp") else ["stl"]


def _smooth_surface(src: Path, opts):
    """Read a topology ``.vtu``, extract its surface and smooth it.

    Returns the pyvista surface mesh. Raises on read/smooth failure; callers
    wrap this so a single bad snapshot never aborts the rest.
    """
    import pyvista as pv
    grid = pv.read(str(src))
    surf = grid.extract_surface(algorithm="dataset_surface").triangulate()
    if str(opts.method).lower() == "laplacian":
        return surf.smooth(n_iter=int(opts.iterations),
                           relaxation_factor=float(opts.relaxation))
    return surf.smooth_taubin(n_iter=int(opts.iterations),
                              pass_band=float(opts.pass_band))


def _save_surface(surf, work: Path, base: str, output_format: str,
                  log: Callable[[str], None]) -> list[Path]:
    """Save *surf* as ``<work>/<base>.<ext>`` for each requested format."""
    written: list[Path] = []
    for ext in _formats(output_format):
        dest = Path(work) / f"{base}.{ext}"
        try:
            surf.save(str(dest))
            written.append(dest)
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] smooth: could not write {dest.name}: {exc}")
    return written


def _pyvista_or_none(log: Callable[[str], None]) -> bool:
    try:
        import pyvista  # noqa: F401
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"[oropt] smooth: pyvista unavailable: {exc} - skipped")
        return False


def smooth_final(cfg: Config, work: Path,
                 log: Callable[[str], None] = print) -> Optional[Path]:
    """If enabled, smooth the final design's surface and write
    ``<work>/topology_smoothed.<ext>``.

    Returns the path of the first file written, else ``None`` (reason logged).
    """
    opts = cfg.smooth
    if not opts.enabled:
        return None
    src = Path(work) / st.TOPOLOGY
    if not src.is_file():
        log(f"[oropt] smooth: no {st.TOPOLOGY} to smooth - skipped")
        return None
    if not _pyvista_or_none(log):
        return None
    try:
        surf = _smooth_surface(src, opts)
    except Exception as exc:  # noqa: BLE001
        log(f"[oropt] smooth: failed to smooth surface: {exc} - skipped")
        return None
    written = _save_surface(surf, work, SMOOTHED_BASE, opts.output_format, log)
    if not written:
        return None
    log(f"[oropt] smooth: wrote {', '.join(p.name for p in written)} "
        f"({surf.n_points} pts, {int(opts.iterations)} {opts.method} passes)")
    return written[0]


def smooth_all_iterations(cfg: Config, work: Path,
                          log: Callable[[str], None] = print) -> list[Path]:
    """If enabled, smooth *every* per-iteration topology snapshot.

    Each ``<work>/topology_iterNNNN.vtu`` is smoothed into
    ``<work>/topology_smoothed_iterNNNN.<ext>``, giving the smoothed shape at
    every iteration alongside the per-iteration raw snapshots. Best-effort: a
    snapshot that fails to smooth is logged and skipped; the rest still run.
    Returns the list of files written (empty when disabled / nothing to do).
    """
    opts = cfg.smooth
    if not opts.enabled:
        return []
    work = Path(work)
    snaps = sorted(work.glob("topology_iter*.vtu"))
    if not snaps:
        log("[oropt] smooth: no per-iteration snapshots to smooth - skipped")
        return []
    if not _pyvista_or_none(log):
        return []
    written_all: list[Path] = []
    for src in snaps:
        # topology_iter0007.vtu -> base topology_smoothed_iter0007
        base = f"{SMOOTHED_BASE}_{src.stem[len('topology_'):]}"
        try:
            surf = _smooth_surface(src, opts)
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] smooth: failed on {src.name}: {exc} - skipped")
            continue
        written_all.extend(_save_surface(surf, work, base, opts.output_format, log))
    if written_all:
        log(f"[oropt] smooth: wrote {len(written_all)} per-iteration smoothed "
            f"file(s) for {len(snaps)} snapshot(s) "
            f"({int(opts.iterations)} {opts.method} passes each)")
    return written_all

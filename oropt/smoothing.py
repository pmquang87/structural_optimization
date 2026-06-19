"""Best-effort surface smoothing of the final optimised geometry.

After a run finishes, the surface of the final design (the latest
``topology_latest.vtu``) is extracted, smoothed and written as
``topology_smoothed.<ext>`` in the run folder — a clean deliverable for
CAD / 3D-print / review. Uses pyvista, which the loop already requires for its
per-iteration topology output. Never raises: failures are logged and return
``None`` so smoothing can never abort or fail an optimisation run.
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
    try:
        import pyvista as pv
    except Exception as exc:  # noqa: BLE001
        log(f"[oropt] smooth: pyvista unavailable: {exc} - skipped")
        return None
    try:
        grid = pv.read(str(src))
        surf = grid.extract_surface(algorithm="dataset_surface").triangulate()
        if str(opts.method).lower() == "laplacian":
            surf = surf.smooth(n_iter=int(opts.iterations),
                               relaxation_factor=float(opts.relaxation))
        else:
            surf = surf.smooth_taubin(n_iter=int(opts.iterations),
                                      pass_band=float(opts.pass_band))
    except Exception as exc:  # noqa: BLE001
        log(f"[oropt] smooth: failed to smooth surface: {exc} - skipped")
        return None
    written: list[Path] = []
    for ext in _formats(opts.output_format):
        dest = Path(work) / f"{SMOOTHED_BASE}.{ext}"
        try:
            surf.save(str(dest))
            written.append(dest)
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] smooth: could not write {dest.name}: {exc}")
    if not written:
        return None
    log(f"[oropt] smooth: wrote {', '.join(p.name for p in written)} "
        f"({surf.n_points} pts, {int(opts.iterations)} {opts.method} passes)")
    return written[0]

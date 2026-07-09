"""Best-effort conversion of OpenRadioss animation files into an LS-Dyna d3plot.

The heavy lifting is done by the external Vortex-Radioss ``Anim_to_D3plot`` tool,
which we invoke in an *isolated subprocess* (its own interpreter + dependencies)
so oropt's environment never needs lasso-python/tqdm. Nothing here raises: every
failure path is logged and returns ``None`` so a conversion problem can never
abort or fail an optimisation run.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from .config import Config, D3plotOpts

# Run by the (external) interpreter: put the tool root on sys.path, import the
# converter and run it against a file stem. argv -> [tool_root, stem, rigidwall].
_RUNNER = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from vortex_radioss.animtod3plot.Anim_to_D3plot import readAndConvert; "
    "readAndConvert(sys.argv[2], use_shell_mask=True, use_solid_mask=False, "
    "use_beam_mask=False, silent=True, no_warnings=False, "
    "show_rigidwall=(sys.argv[3] == '1'))"
)


def _resolve_tool_root(opts: D3plotOpts) -> str:
    """Folder holding the ``vortex_radioss`` package and (optionally) its ``.venv``.

    An explicit ``tool_root`` wins; otherwise fall back to the ``OROPT_VORTEX_ROOT``
    environment variable so the shipped default config carries no hard-coded
    per-user path. Blank when neither is set (the caller then logs and skips).
    """
    if opts.tool_root.strip():
        return opts.tool_root.strip()
    return os.environ.get("OROPT_VORTEX_ROOT", "").strip()


def _resolve_python(opts: D3plotOpts) -> str:
    """Interpreter to run the converter with (one that has lasso-python/tqdm).

    An explicit ``python_exe`` wins; otherwise prefer the tool root's own
    ``.venv`` (where the converter's deps live), falling back to the interpreter
    currently running oropt.
    """
    if opts.python_exe.strip():
        return opts.python_exe.strip()
    tool_root = _resolve_tool_root(opts)
    if tool_root:
        # Probe both venv layouts -- Scripts/python.exe is Windows-only; a
        # provisioned POSIX venv (bin/python) was silently skipped, dropping
        # to sys.executable, which by design lacks the converter's deps.
        for rel in (("Scripts", "python.exe"), ("bin", "python")):
            venv = Path(tool_root, ".venv", *rel)
            if venv.is_file():
                return str(venv)
    return sys.executable


def convert_stem(stem_path: Path, opts: D3plotOpts,
                 log: Callable[[str], None] = print) -> Optional[Path]:
    """Convert anim files ``<stem_path>A0*`` into ``<stem_path>.d3plot`` in place.

    Returns the written ``.d3plot`` path, or ``None`` (with the reason logged)
    when there is nothing to convert, the interpreter/tool is missing, or the
    subprocess fails or times out.
    """
    stem_path = Path(stem_path)
    if not sorted(stem_path.parent.glob(stem_path.name + "A0*")):
        log(f"[oropt] d3plot: no animation files at {stem_path}A0* - skipped")
        return None
    py = _resolve_python(opts)
    if not Path(py).is_file():
        log(f"[oropt] d3plot: converter interpreter not found: {py} - skipped")
        return None
    tool_root = _resolve_tool_root(opts)
    if not tool_root or not (Path(tool_root) / "vortex_radioss").is_dir():
        log(f"[oropt] d3plot: no vortex_radioss package under tool root "
            f"{tool_root or '(unset; set d3plot.tool_root or OROPT_VORTEX_ROOT)'} "
            "- skipped")
        return None
    cmd = [py, "-c", _RUNNER, str(tool_root), str(stem_path),
           "1" if opts.show_rigidwall else "0"]
    log(f"[oropt] d3plot: converting {stem_path.name}A0* via {Path(py).name} ...")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=opts.timeout_s)
    except subprocess.TimeoutExpired:
        log(f"[oropt] d3plot: timed out after {opts.timeout_s:.0f}s - skipped")
        return None
    except OSError as exc:
        log(f"[oropt] d3plot: could not launch converter: {exc} - skipped")
        return None
    out = Path(str(stem_path) + ".d3plot")
    if proc.returncode == 0 and out.is_file():
        return out
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    log(f"[oropt] d3plot: conversion failed (rc={proc.returncode}): "
        f"{detail[-1] if detail else 'no output'} - skipped")
    return None


def convert_final(cfg: Config, solve_dir: Path, work: Path,
                  stem: Optional[str] = None,
                  log: Callable[[str], None] = print) -> Optional[Path]:
    """If enabled, convert the final design's animation (in ``solve_dir``) and
    move the resulting ``<stem>.d3plot*`` files into ``<work>/d3plot/``.

    *stem* selects which case's animation to convert; the loop passes each case's
    stem. When omitted it defaults to the primary (first) load case's stem.
    Returns the path of the moved ``<stem>.d3plot`` on success, else ``None``.
    """
    opts = cfg.d3plot
    if not opts.enabled:
        return None
    stem = stem if stem is not None else cfg.primary_case().stem
    if convert_stem(solve_dir / stem, opts, log) is None:
        return None
    dest_dir = work / "d3plot"
    dest_dir.mkdir(parents=True, exist_ok=True)
    result: Optional[Path] = None
    for src in sorted(solve_dir.glob(stem + ".d3plot*")):
        dest = dest_dir / src.name
        dest.unlink(missing_ok=True)
        try:
            shutil.move(str(src), str(dest))
        except OSError as exc:
            log(f"[oropt] d3plot: could not move {src.name} -> {dest_dir}: {exc}")
            continue
        if src.name == stem + ".d3plot":
            result = dest
    log(f"[oropt] d3plot: result written to {dest_dir}")
    return result

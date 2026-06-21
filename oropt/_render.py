"""Run an off-screen pyvista render in an *isolated subprocess*.

The report's final-topology image and the evolution animation's frames are both
drawn off-screen with pyvista/VTK. On a headless box a missing/old GL stack can
*hard-crash* (segfault) the renderer — uncatchable in-process — so each render is
run in a throwaway subprocess (the same interpreter, which already has pyvista).
A crash can then only surface as a non-zero exit code, never as an aborted run.

This module owns that one contract — launch a ``python -c <script> <args...>``
render with a timeout and classify the outcome — so :mod:`oropt.report` and
:mod:`oropt.animate` share it instead of each re-implementing the
launch/timeout/return-code handling (which must stay identical for the crash
isolation the CI relies on).
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RenderResult:
    """Outcome of an isolated render subprocess.

    ``ok`` is True only on a clean exit. On failure *detail* is a short
    human-readable reason (the last line of the child's output, or why it could
    not run); *returncode* is ``None`` when the child never produced one (timeout
    or launch failure) and the process exit code otherwise.
    """
    ok: bool
    returncode: Optional[int]
    detail: str

    @property
    def timed_out(self) -> bool:
        return not self.ok and self.returncode is None and "timed out" in self.detail


def run_render(script: str, args, timeout_s: float) -> RenderResult:
    """Run ``python -c script <args...>`` off-screen, capped at *timeout_s*.

    Returns a :class:`RenderResult`; never raises for a render failure (a missing
    interpreter, a timeout or a VTK/GL crash all come back as ``ok=False`` with a
    reason). The caller decides how to log and degrade.
    """
    cmd = [sys.executable, "-c", script, *(str(a) for a in args)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=float(timeout_s))
    except subprocess.TimeoutExpired:
        return RenderResult(False, None, f"timed out after {timeout_s:.0f}s")
    except OSError as exc:
        return RenderResult(False, None, f"could not launch renderer: {exc}")
    if proc.returncode == 0:
        return RenderResult(True, 0, "")
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return RenderResult(False, proc.returncode,
                        detail[-1] if detail else "no output")

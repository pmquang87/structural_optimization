"""Launch OpenRadioss (starter + engine) as a subprocess, np=1.

Ports the working ``run_or.ps1`` recipe to Python: the pure-OpenMP engine
(``engine_win64.exe``, no mpiexec) is used because the model must run with a
single MPI domain (SPMD implicit + solid contact segfaults). Environment mirrors
the PowerShell launcher plus the i9-13900H livelock mitigation
(``KMP_BLOCKTIME=0`` / ``OMP_WAIT_POLICY=PASSIVE`` / 6 threads).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config

# PATH entries the win64 binaries need (hm_reader DLL for the starter,
# libiomp5md.dll for the OpenMP engine, h3d libs for animation output).
_PATH_SUBDIRS = (
    "exec",
    r"extlib\hm_reader\win64",
    r"extlib\h3d\lib\win64",
    r"extlib\intelOneAPI_runtime\win64",
)


@dataclass
class RunResult:
    ok: bool
    stage: str                 # "starter" | "engine" | "ok"
    message: str
    returncode: Optional[int] = None
    cycles: Optional[int] = None
    elapsed_s: Optional[float] = None
    # True when the engine was killed because the implicit solve was judged
    # non-converging (a diverge-cycle streak or the soft wall-clock budget) --
    # NOT a solver error. The loop treats such an iteration as INFEASIBLE and
    # backs off instead of failing the run.
    diverged: bool = False


def build_env(cfg: Config) -> dict:
    """Environment for the OpenRadioss subprocess (a copy of os.environ + OR vars)."""
    p = cfg.or_paths
    env = dict(os.environ)
    env["OPENRADIOSS_PATH"] = str(Path(p.root))
    env["RAD_CFG_PATH"] = str(p.abs("cfg_path"))
    env["RAD_H3D_PATH"] = str(p.abs("h3d_path"))
    env["KMP_STACKSIZE"] = cfg.run.kmp_stacksize
    env["OMP_NUM_THREADS"] = str(cfg.run.nt)
    env["KMP_AFFINITY"] = "disabled"
    env["KMP_BLOCKTIME"] = "0"            # livelock mitigation
    env["OMP_WAIT_POLICY"] = "PASSIVE"    # livelock mitigation
    prepend = [str(Path(p.root) / sub) for sub in _PATH_SUBDIRS]
    if cfg.run.use_mpi:                   # Intel MPI runtime (impi.dll, libfabric) for the engine
        env["I_MPI_ROOT"] = str(Path(p.intel_mpi_root))
        env["I_MPI_OFI_LIBRARY_INTERNAL"] = "1"
        prepend = p.mpi_path_dirs() + prepend
    env["PATH"] = os.pathsep.join(prepend + [env.get("PATH", "")])
    return env


def backend_problems(cfg: Config) -> list[str]:
    """Solver executables/CLI that are required but missing on this machine.

    The single source of truth shared by :func:`run_solver`'s pre-flight and
    :func:`oropt.validate.validate_config`, so the fast config check and the real
    run agree on what "the backend is installed" means. Returns one human-readable
    message per missing piece; an empty list means the backend is ready.

    Docker backend -> the ``docker`` CLI must resolve; native backend -> the
    OpenRadioss starter & engine (and ``mpiexec`` when ``run.use_mpi``) must exist.
    Demo backend (``demo.enabled``) -> nothing: the synthetic physics needs no
    solver at all, so both callers stay quiet.
    """
    problems: list[str] = []
    if getattr(cfg, "demo", None) is not None and cfg.demo.enabled:
        return problems
    if cfg.docker.enabled:
        exe = cfg.docker.docker_exe
        if shutil.which(exe) is None and not Path(exe).exists():
            problems.append(f"docker CLI not found: {exe} "
                            "(install/start Docker Desktop, or set docker.docker_exe)")
    else:
        for attr in ("starter", "engine"):
            exe_path = cfg.or_paths.abs(attr)
            if not exe_path.exists():
                problems.append(f"executable not found: {exe_path}")
        if cfg.run.use_mpi and not cfg.or_paths.mpiexec().exists():
            problems.append(f"mpiexec not found: {cfg.or_paths.mpiexec()}")
    return problems


def _docker_base(cfg: Config, run_dir: Path,
                 cidfile: Optional[Path] = None) -> list[str]:
    """``docker run`` prefix that bind-mounts *run_dir* to ``/data`` (forward-slash
    path so Docker Desktop accepts the Windows drive), up to the image name.

    *cidfile* records the container id on the host: killing the docker CLI
    client only detaches it — the container keeps solving (``--rm`` reaps only
    after the container *exits* on its own) — so every kill path needs the id to
    ``docker kill`` the container itself (see :func:`_kill_container`)."""
    d = cfg.docker
    data = str(Path(run_dir).resolve()).replace("\\", "/")
    cid = ["--cidfile", str(cidfile)] if cidfile is not None else []
    return [d.docker_exe, "run", "--rm", f"--shm-size={d.shm_size}", *cid,
            "-v", f"{data}:/data", "-w", "/data", *list(d.extra_args), d.image]


def _kill_container(cfg: Config, cidfile: Path) -> None:
    """Best-effort ``docker kill`` of the container recorded in *cidfile*.

    Without this, every diverged / soft-timeout "kill" leaves a zombie container
    grinding at full thread count: a few diverged iterations in a row saturate
    the machine, slow the *live* solve past its soft budget, and cascade into
    ``diverge_fail_after`` failing the whole run."""
    try:
        cid = cidfile.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if cid:
        try:
            subprocess.run([cfg.docker.docker_exe, "kill", cid],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
    cidfile.unlink(missing_ok=True)


def _cidfile(run_dir: Path, stem: str, stage: str) -> Optional[Path]:
    """Host path recording the *stage* container's id (fresh per launch: docker
    refuses to start when the cidfile already exists)."""
    p = Path(run_dir) / f"{stem}_{stage}.cid"
    p.unlink(missing_ok=True)
    return p


def _starter_cmd(cfg: Config, run_dir: Path, stem: str,
                 cidfile: Optional[Path] = None) -> list[str]:
    if cfg.docker.enabled:
        d = cfg.docker
        return _docker_base(cfg, run_dir, cidfile) + [
            "starter", "-i", f"{stem}_0000.rad", "-np", str(d.np), "-nt", str(d.nt)]
    return [str(cfg.or_paths.abs("starter")), "-i", f"{stem}_0000.rad",
            "-np", str(cfg.run.np)]


def _engine_cmd(cfg: Config, run_dir: Path, stem: str,
                cidfile: Optional[Path] = None) -> list[str]:
    if cfg.docker.enabled:
        d = cfg.docker
        return _docker_base(cfg, run_dir, cidfile) + [
            "engine", str(d.np), "-i", f"{stem}_0001.rad", "-nt", str(d.nt)]
    engine = cfg.or_paths.abs("engine")
    if cfg.run.use_mpi:        # np=1 via mpiexec; the bare engine cannot load its MPI DLLs
        return [str(cfg.or_paths.mpiexec()), "-np", str(cfg.run.np),
                str(engine), "-i", f"{stem}_0001.rad", "-nt", str(cfg.run.nt)]
    return [str(engine), "-i", f"{stem}_0001.rad", "-nt", str(cfg.run.nt)]


def _log_tail(path: Path, n: int = 12) -> str:
    """Last *n* non-blank lines of a log, joined — for surfacing docker errors."""
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return " | ".join(s.strip() for s in lines[-n:] if s.strip())


def _run(cmd, cwd: Path, env: dict, log: Path, timeout: float) -> subprocess.CompletedProcess:
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        return subprocess.run(
            cmd, cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )


def _starter_ok(out_file: Path) -> tuple[bool, str]:
    """Starter success = its listing reports ``0 ERROR(S)`` (it never prints
    "NORMAL TERMINATION" — that is the engine's marker)."""
    if not out_file.exists():
        return False, f"missing starter listing: {out_file.name}"
    text = out_file.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(\d+)\s+ERROR\(S\)", text)
    if m:
        n = int(m.group(1))
        return (n == 0), ("0 ERROR(S)" if n == 0 else f"{n} starter ERROR(S)")
    return False, "starter ended without an ERROR(S) summary"


def _engine_ok(out_file: Path) -> tuple[bool, str]:
    if not out_file.exists():
        return False, f"missing engine listing: {out_file.name}"
    text = out_file.read_text(encoding="utf-8", errors="replace")
    if "NORMAL TERMINATION" in text:
        return True, "NORMAL TERMINATION"
    for marker in ("ERROR TERMINATION", "ERROR ID", "** ERROR"):
        if marker in text:
            for line in text.splitlines():
                if marker in line:
                    return False, line.strip()
    return False, "no NORMAL TERMINATION found"


# ---- implicit non-convergence watchdog --------------------------------------
# A design whose load path was severed (e.g. over-carved by the optimiser) does
# not make the implicit engine error out: every attempt prints
# "--ITERATION DIVERGE with MAX_ITER REACHED--" then
# "--NEXT TIMESTEP IS DECREASED BY-- 0.6667E+00" and the retry diverges again,
# grinding until engine_timeout_s (typically hours). Healthy solves print
# *isolated* DIVERGE lines too and recover -- the next step converges (an
# iteration row ending in "C") and the timestep is increased back -- so only an
# unbroken streak of diverge cycles with no accepted step between them counts.

_DIVERGE_MARK = "--ITERATION DIVERGE"
_DT_INCREASE_MARK = "IS INCREASED BY"
# An accepted iteration row of the implicit convergence table, e.g.
# "     2              4.929E-01  4.653E-02  1.835E-02     C"
# (iter number, optional stiffness-reformed "Y", three residuals, Conv.stat C).
_CONV_ROW = re.compile(
    r"^\s*\d+\s+(?:Y\s+)?(?:[0-9][0-9.eE+-]*\s+){3}C\s*$")

_POLL_S = 5.0   # engine watchdog poll interval (module-level so tests can shrink it)


class DivergenceMonitor:
    """Streaming detector for a non-converging implicit solve.

    Feed it chunks of the growing engine listing (any split, mid-line is fine);
    it tracks the current streak of consecutive ITERATION DIVERGE cycles,
    resetting whenever a step is accepted (a converged iteration row or a
    timestep increase). Returns a reason string once the streak reaches
    *max_cycles*, ``None`` while the solve still looks alive. The check runs
    after each whole chunk so a streak that already recovered within the same
    chunk never trips. ``max_cycles <= 0`` disables detection."""

    def __init__(self, max_cycles: int):
        self.max_cycles = int(max_cycles)
        self.streak = 0
        self._tail = ""          # unfinished last line, waits for the next chunk

    def feed(self, chunk: str) -> Optional[str]:
        if self.max_cycles <= 0:
            return None
        if chunk:
            lines = (self._tail + chunk).split("\n")
            self._tail = lines.pop()
            for line in lines:
                if _DIVERGE_MARK in line:
                    self.streak += 1
                elif _DT_INCREASE_MARK in line or _CONV_ROW.match(line):
                    self.streak = 0
        if self.streak >= self.max_cycles:
            return (f"{self.streak} consecutive ITERATION DIVERGE / "
                    "timestep-decrease cycles without an accepted step "
                    f"(run.diverge_max_cycles={self.max_cycles})")
        return None


def _read_new(path: Path, offset: list) -> str:
    """New bytes of *path* since ``offset[0]`` (advanced in place); "" while the
    file does not exist yet or nothing was appended."""
    try:
        with open(path, "rb") as fh:
            fh.seek(offset[0])
            data = fh.read()
            offset[0] += len(data)
    except OSError:
        return ""
    return data.decode("utf-8", "replace")


def _run_engine(cfg: Config, cmd, cwd: Path, env: dict, log: Path,
                listing: Path, cidfile: Optional[Path] = None
                ) -> tuple[Optional[int], Optional[str]]:
    """Run the engine under the non-convergence watchdog.

    Like :func:`_run` but polls the growing listing (and the console log)
    every ``_POLL_S`` seconds for a diverge-cycle streak, and enforces the soft
    wall-clock budget ``run.engine_soft_timeout_s``. Returns ``(returncode,
    None)`` when the engine exits on its own, or ``(None, reason)`` after
    killing a solve judged non-converging. ``run.engine_timeout_s`` stays the
    hard kill and raises :class:`subprocess.TimeoutExpired`, exactly like the
    plain runner. On the docker backend killing the CLI client only detaches
    the container, so every kill path also ``docker kill``s the container via
    *cidfile* (see :func:`_kill_container`)."""
    soft = cfg.run.engine_soft_timeout_s
    hard = cfg.run.engine_timeout_s
    # Fresh watch state; drop a stale listing from a previous solve in the same
    # dir so an old diverge streak cannot trip the monitor before the engine
    # recreates the file.
    listing.unlink(missing_ok=True)
    watch = [(listing, DivergenceMonitor(cfg.run.diverge_max_cycles), [0]),
             (log, DivergenceMonitor(cfg.run.diverge_max_cycles), [0])]
    start = time.monotonic()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=fh,
                                stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    return proc.wait(timeout=_POLL_S), None
                except subprocess.TimeoutExpired:
                    pass
                elapsed = time.monotonic() - start
                if elapsed >= hard:
                    raise subprocess.TimeoutExpired(cmd, hard)
                if soft and soft > 0 and elapsed >= soft:
                    return None, ("wall clock exceeded run.engine_soft_timeout_s"
                                  f" ({soft:.0f} s)")
                for path, monitor, offset in watch:
                    reason = monitor.feed(_read_new(path, offset))
                    if reason:
                        return None, reason
        finally:
            if proc.poll() is None:
                proc.kill()
                if cidfile is not None:      # the container outlives its client
                    _kill_container(cfg, cidfile)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            elif cidfile is not None:        # clean exit: just drop the id file
                cidfile.unlink(missing_ok=True)


def _parse_engine_stats(out_file: Path) -> tuple[Optional[int], Optional[float]]:
    if not out_file.exists():
        return None, None
    text = out_file.read_text(encoding="utf-8", errors="replace")
    cyc = re.search(r"TOTAL NUMBER OF CYCLES\s*:\s*(\d+)", text)
    el = re.search(r"ELAPSED TIME\s*=\s*([\d.]+)\s*s", text)
    return (int(cyc.group(1)) if cyc else None,
            float(el.group(1)) if el else None)


def run_solver(cfg: Config, run_dir: str | Path,
               stem: Optional[str] = None) -> RunResult:
    """Run starter then engine in *run_dir*; return a RunResult.

    The deck (``<stem>_0000.rad`` / ``_0001.rad``) must already exist in *run_dir*.
    *stem* selects which deck to solve; the loop passes each case's stem. When
    omitted it defaults to the primary (first) load case's stem.
    """
    run_dir = Path(run_dir).resolve()
    stem = stem if stem is not None else cfg.primary_case().stem

    # --- backend pre-flight + environment ---
    problems = backend_problems(cfg)
    if problems:
        return RunResult(False, "setup", "; ".join(problems))
    # the container carries its own OR runtime; native needs the OR env vars/PATH
    env = dict(os.environ) if cfg.docker.enabled else build_env(cfg)

    # --- starter ---
    starter_log = run_dir / f"{stem}_starter.log"
    starter_cid = _cidfile(run_dir, stem, "starter") if cfg.docker.enabled else None
    try:
        cp = _run(_starter_cmd(cfg, run_dir, stem, starter_cid), run_dir, env,
                  starter_log, cfg.run.starter_timeout_s)
    except subprocess.TimeoutExpired:
        if starter_cid is not None:          # the container outlives its client
            _kill_container(cfg, starter_cid)
        return RunResult(False, "starter", "starter timed out", cycles=None)
    if starter_cid is not None:
        starter_cid.unlink(missing_ok=True)
    ok, msg = _starter_ok(run_dir / f"{stem}_0000.out")
    if not ok:
        if cfg.docker.enabled and cp.returncode != 0:   # surface docker/daemon errors
            msg = f"{msg} [docker rc={cp.returncode}: {_log_tail(starter_log)}]"
        return RunResult(False, "starter", f"starter failed: {msg}", cp.returncode)

    # --- engine (under the non-convergence watchdog) ---
    engine_log = run_dir / f"{stem}_engine.log"
    listing = run_dir / f"{stem}_0001.out"
    engine_cid = _cidfile(run_dir, stem, "engine") if cfg.docker.enabled else None
    try:
        rc, diverged = _run_engine(cfg, _engine_cmd(cfg, run_dir, stem, engine_cid),
                                   run_dir, env, engine_log, listing,
                                   cidfile=engine_cid)
    except subprocess.TimeoutExpired:
        return RunResult(False, "engine", "engine timed out")
    cycles, elapsed = _parse_engine_stats(listing)
    if diverged:
        return RunResult(False, "engine", f"engine did not converge: {diverged}",
                         None, cycles, elapsed, diverged=True)
    ok, msg = _engine_ok(listing)
    if not ok:
        if cfg.docker.enabled and rc != 0:
            msg = f"{msg} [docker rc={rc}: {_log_tail(engine_log)}]"
        return RunResult(False, "engine", f"engine failed: {msg}", rc, cycles, elapsed)
    return RunResult(True, "ok", msg, rc, cycles, elapsed)


def find_t01(run_dir: str | Path, stem: str) -> Optional[Path]:
    p = Path(run_dir) / f"{stem}T01"
    return p if p.exists() else None


def find_last_anim(run_dir: str | Path, stem: str) -> Optional[Path]:
    """Latest animation file ``<stem>A0NN`` in *run_dir* (highest index).

    Sorted numerically on the index: OpenRadioss widens the number past
    ``A999`` to ``A1000``, and lexicographically ``A1000 < A999`` — a plain
    sort would silently extract a mid-run state as the "final" one.
    """
    run_dir = Path(run_dir)

    def _idx(p: Path) -> int:
        m = re.search(r"A(\d+)$", p.name)
        return int(m.group(1)) if m else -1

    anims = [p for p in run_dir.glob(f"{stem}A[0-9][0-9]*") if _idx(p) >= 0]
    return max(anims, key=_idx) if anims else None

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
    """
    problems: list[str] = []
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


def _docker_base(cfg: Config, run_dir: Path) -> list[str]:
    """``docker run`` prefix that bind-mounts *run_dir* to ``/data`` (forward-slash
    path so Docker Desktop accepts the Windows drive), up to the image name."""
    d = cfg.docker
    data = str(Path(run_dir).resolve()).replace("\\", "/")
    return [d.docker_exe, "run", "--rm", f"--shm-size={d.shm_size}",
            "-v", f"{data}:/data", "-w", "/data", *list(d.extra_args), d.image]


def _starter_cmd(cfg: Config, run_dir: Path, stem: str) -> list[str]:
    if cfg.docker.enabled:
        d = cfg.docker
        return _docker_base(cfg, run_dir) + [
            "starter", "-i", f"{stem}_0000.rad", "-np", str(d.np), "-nt", str(d.nt)]
    return [str(cfg.or_paths.abs("starter")), "-i", f"{stem}_0000.rad",
            "-np", str(cfg.run.np)]


def _engine_cmd(cfg: Config, run_dir: Path, stem: str) -> list[str]:
    if cfg.docker.enabled:
        d = cfg.docker
        return _docker_base(cfg, run_dir) + [
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


def _parse_engine_stats(out_file: Path) -> tuple[Optional[int], Optional[float]]:
    if not out_file.exists():
        return None, None
    text = out_file.read_text(encoding="utf-8", errors="replace")
    cyc = re.search(r"TOTAL NUMBER OF CYCLES\s*:\s*(\d+)", text)
    el = re.search(r"ELAPSED TIME\s*=\s*([\d.]+)\s*s", text)
    return (int(cyc.group(1)) if cyc else None,
            float(el.group(1)) if el else None)


def run_solver(cfg: Config, run_dir: str | Path) -> RunResult:
    """Run starter then engine in *run_dir*; return a RunResult.

    The deck (``<stem>_0000.rad`` / ``_0001.rad``) must already exist in *run_dir*.
    """
    run_dir = Path(run_dir).resolve()
    stem = cfg.model.stem

    # --- backend pre-flight + environment ---
    problems = backend_problems(cfg)
    if problems:
        return RunResult(False, "setup", "; ".join(problems))
    # the container carries its own OR runtime; native needs the OR env vars/PATH
    env = dict(os.environ) if cfg.docker.enabled else build_env(cfg)

    # --- starter ---
    starter_log = run_dir / f"{stem}_starter.log"
    try:
        cp = _run(_starter_cmd(cfg, run_dir, stem), run_dir, env,
                  starter_log, cfg.run.starter_timeout_s)
    except subprocess.TimeoutExpired:
        return RunResult(False, "starter", "starter timed out", cycles=None)
    ok, msg = _starter_ok(run_dir / f"{stem}_0000.out")
    if not ok:
        if cfg.docker.enabled and cp.returncode != 0:   # surface docker/daemon errors
            msg = f"{msg} [docker rc={cp.returncode}: {_log_tail(starter_log)}]"
        return RunResult(False, "starter", f"starter failed: {msg}", cp.returncode)

    # --- engine ---
    engine_log = run_dir / f"{stem}_engine.log"
    try:
        cp = _run(_engine_cmd(cfg, run_dir, stem), run_dir, env,
                  engine_log, cfg.run.engine_timeout_s)
    except subprocess.TimeoutExpired:
        return RunResult(False, "engine", "engine timed out")
    ok, msg = _engine_ok(run_dir / f"{stem}_0001.out")
    cycles, elapsed = _parse_engine_stats(run_dir / f"{stem}_0001.out")
    if not ok:
        if cfg.docker.enabled and cp.returncode != 0:
            msg = f"{msg} [docker rc={cp.returncode}: {_log_tail(engine_log)}]"
        return RunResult(False, "engine", f"engine failed: {msg}", cp.returncode, cycles, elapsed)
    return RunResult(True, "ok", msg, cp.returncode, cycles, elapsed)


def find_t01(run_dir: str | Path, stem: str) -> Optional[Path]:
    p = Path(run_dir) / f"{stem}T01"
    return p if p.exists() else None


def find_last_anim(run_dir: str | Path, stem: str) -> Optional[Path]:
    """Latest animation file ``<stem>A0NN`` in *run_dir* (highest index)."""
    run_dir = Path(run_dir)
    anims = sorted(run_dir.glob(f"{stem}A[0-9][0-9]*"))
    return anims[-1] if anims else None

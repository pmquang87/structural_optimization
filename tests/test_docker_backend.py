"""Docker solver backend: config plumbing + command construction.

These build and inspect the command lists only — they never invoke Docker or the
native solver, so they stay fast and hermetic.
"""
from __future__ import annotations

import os
from pathlib import Path

from oropt.config import Config
from oropt.runner import _docker_base, _engine_cmd, _starter_cmd


def test_docker_defaults_off_and_roundtrip(tmp_path):
    cfg = Config()
    assert cfg.docker.enabled is False
    assert cfg.docker.image == "openradioss-mumps:20260520"
    cfg.docker.enabled = True
    cfg.docker.np = 8
    cfg.docker.extra_args = ["--cpus", "8"]
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.docker.enabled is True
    assert back.docker.np == 8
    assert back.docker.extra_args == ["--cpus", "8"]


def test_native_starter_cmd_uses_local_exe():
    cfg = Config()                                  # docker disabled
    cmd = _starter_cmd(cfg, Path("/run"), "demo")
    assert cmd[-4:] == ["-i", "demo_0000.rad", "-np", str(cfg.run.np)]
    assert "docker" not in cmd[0].lower()


def test_native_engine_cmd_mpi():
    cfg = Config()                                  # use_mpi True, docker off
    cmd = _engine_cmd(cfg, Path("/run"), "demo")
    assert "-np" in cmd and "demo_0001.rad" in cmd and "-nt" in cmd


def test_docker_starter_cmd_structure(tmp_path):
    cfg = Config()
    cfg.docker.enabled = True
    cfg.docker.np, cfg.docker.nt = 4, 1
    cmd = _starter_cmd(cfg, tmp_path, "demo")
    assert cmd[0] == "docker" and "run" in cmd and "--rm" in cmd
    assert "--shm-size=2g" in cmd
    vi = cmd.index("-v")                             # bind mount: forward slashes -> /data
    assert cmd[vi + 1].endswith(":/data") and "\\" not in cmd[vi + 1]
    assert cfg.docker.image in cmd
    assert cmd[cmd.index("starter"):] == [
        "starter", "-i", "demo_0000.rad", "-np", "4", "-nt", "1"]


def test_docker_engine_cmd_np_is_positional(tmp_path):
    cfg = Config()
    cfg.docker.enabled = True
    cfg.docker.np, cfg.docker.nt = 4, 2
    cmd = _engine_cmd(cfg, tmp_path, "demo")
    assert cmd[cmd.index("engine"):] == [
        "engine", "4", "-i", "demo_0001.rad", "-nt", "2"]


def test_docker_extra_args_before_image(tmp_path):
    cfg = Config()
    cfg.docker.enabled = True
    cfg.docker.extra_args = ["--cpus", "8"]
    base = _docker_base(cfg, tmp_path)
    assert "--cpus" in base and "8" in base
    assert base.index("--cpus") < base.index(cfg.docker.image)


# ---- container kill (a killed docker CLI leaves the container solving) -------
def test_docker_cmds_carry_cidfile(tmp_path):
    from oropt.runner import _cidfile
    cfg = Config()
    cfg.docker.enabled = True
    cid = _cidfile(tmp_path, "demo", "engine")
    cmd = _engine_cmd(cfg, tmp_path, "demo", cid)
    i = cmd.index("--cidfile")
    assert cmd[i + 1] == str(cid)
    assert str(cid).endswith("demo_engine.cid")
    # native path: no cidfile args ever
    cfg.docker.enabled = False
    assert "--cidfile" not in _engine_cmd(cfg, tmp_path, "demo")


def test_cidfile_removes_stale_file(tmp_path):
    """docker refuses to start when the cidfile already exists -- a leftover
    from a previous (killed) solve must not block the next launch."""
    from oropt.runner import _cidfile
    stale = tmp_path / "demo_engine.cid"
    stale.write_text("deadbeef", encoding="utf-8")
    p = _cidfile(tmp_path, "demo", "engine")
    assert p == stale and not stale.exists()


def test_kill_container_invokes_docker_kill(tmp_path, monkeypatch):
    """Killing a non-converging solve must docker-kill the CONTAINER, not just
    the CLI client: --rm reaps only after the container exits on its own, so a
    watchdog 'kill' otherwise leaves a zombie solver grinding at full threads."""
    import subprocess as sp

    from oropt import runner as runner_mod

    calls = []
    monkeypatch.setattr(
        runner_mod.subprocess, "run",
        lambda cmd, **kw: calls.append(cmd) or sp.CompletedProcess(cmd, 0))
    cfg = Config()
    cfg.docker.enabled = True
    cid = tmp_path / "demo_engine.cid"
    cid.write_text("abc123\n", encoding="utf-8")

    runner_mod._kill_container(cfg, cid)

    assert calls == [[cfg.docker.docker_exe, "kill", "abc123"]]
    assert not cid.exists()                      # id file cleaned up


def test_kill_container_noop_without_cidfile(tmp_path, monkeypatch):
    from oropt import runner as runner_mod
    calls = []
    monkeypatch.setattr(runner_mod.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd))
    cfg = Config()
    cfg.docker.enabled = True
    runner_mod._kill_container(cfg, tmp_path / "absent.cid")
    assert calls == []                           # nothing to kill, no CLI call


def test_watchdog_kill_reaps_container(tmp_path, monkeypatch):
    """End-to-end through _run_engine: a diverging docker solve is killed AND
    its container id is docker-killed."""
    import sys as _sys

    from oropt import runner as runner_mod

    cfg = Config()
    cfg.docker.enabled = True
    cfg.run.diverge_max_cycles = 2
    cfg.run.engine_timeout_s = 60.0
    monkeypatch.setattr(runner_mod, "_POLL_S", 0.05)

    killed = []
    monkeypatch.setattr(runner_mod, "_kill_container",
                        lambda c, p: killed.append(p))

    listing = tmp_path / "m_0001.out"
    log = tmp_path / "m_engine.log"
    cid = tmp_path / "m_engine.cid"
    cid.write_text("abc123", encoding="utf-8")
    # a stand-in "engine" that streams diverge cycles forever
    code = ("import time\n"
            "while True:\n"
            "    print('--ITERATION DIVERGE with MAX_ITER REACHED--', flush=True)\n"
            "    time.sleep(0.02)\n")
    rc, reason = runner_mod._run_engine(
        cfg, [_sys.executable, "-u", "-c", code], tmp_path, dict(os.environ),
        log, listing, cidfile=cid)

    assert rc is None and "DIVERGE" in reason
    assert killed == [cid]                       # container reaped, not orphaned

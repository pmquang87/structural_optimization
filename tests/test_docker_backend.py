"""Docker solver backend: config plumbing + command construction.

These build and inspect the command lists only — they never invoke Docker or the
native solver, so they stay fast and hermetic.
"""
from __future__ import annotations

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

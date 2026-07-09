"""Anim->d3plot conversion: config plumbing + best-effort guard paths.

These exercise only the cheap skip/guard branches; they never launch the
external converter (which needs lasso-python), so they stay fast and hermetic.
"""
from __future__ import annotations

import sys

from oropt.config import Config, D3plotOpts
from oropt.d3plot import _resolve_python, convert_final, convert_stem


def test_d3plot_defaults_on_and_roundtrips(tmp_path):
    cfg = Config()
    assert cfg.d3plot.enabled is True                  # on by default
    cfg.d3plot.enabled = False
    cfg.d3plot.tool_root = r"X:\tools\openradioss_tools"
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.d3plot.enabled is False
    assert back.d3plot.tool_root == r"X:\tools\openradioss_tools"


def test_from_dict_ignores_unknown_d3plot_keys():
    cfg = Config.from_dict({"d3plot": {"enabled": True, "bogus": 123}})
    assert cfg.d3plot.enabled is True


def test_resolve_python_prefers_explicit():
    opts = D3plotOpts(python_exe=r"C:\pythons\py.exe")
    assert _resolve_python(opts) == r"C:\pythons\py.exe"


def test_resolve_python_falls_back_to_tool_venv(tmp_path):
    py = tmp_path / ".venv" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    assert _resolve_python(D3plotOpts(tool_root=str(tmp_path))) == str(py)


def test_resolve_python_defaults_to_current_interpreter(tmp_path):
    # tmp_path has no .venv -> fall back to the running interpreter.
    assert _resolve_python(D3plotOpts(tool_root=str(tmp_path))) == sys.executable


def test_convert_stem_no_anim_returns_none(tmp_path):
    logs: list[str] = []
    assert convert_stem(tmp_path / "model", D3plotOpts(), logs.append) is None
    assert any("no animation files" in m for m in logs)


def test_convert_stem_missing_interpreter_returns_none(tmp_path):
    (tmp_path / "modelA001").write_text("x", encoding="utf-8")
    opts = D3plotOpts(python_exe=str(tmp_path / "nope.exe"))
    logs: list[str] = []
    assert convert_stem(tmp_path / "model", opts, logs.append) is None
    assert any("interpreter not found" in m for m in logs)


def test_convert_final_disabled_is_noop(tmp_path):
    cfg = Config()
    cfg.d3plot.enabled = False
    assert convert_final(cfg, tmp_path, tmp_path, lambda *_: None) is None


def test_convert_final_uses_passed_case_stem_multiload(tmp_path):
    """The deck stem lives on the load case; the loop passes each case's stem to
    convert_final, which must look for ``<stem>A0*`` (not a blank-stem ``A0*``
    that would find no animation)."""
    cfg = Config.from_dict({
        "d3plot": {"enabled": True},
        "load_cases": [{"name": "pull", "stem": "implicit_elevator-linkage_pull",
                        "sigma_allow": 300.0, "d_allow": 1.0}],
    })
    primary = cfg.load_case_list()[0]

    solve_dir = tmp_path / "case_0"
    solve_dir.mkdir()
    logs: list[str] = []
    # No animation present -> guard path returns None, but the logged path proves
    # it targeted the passed case's <stem>A0* glob.
    assert convert_final(cfg, solve_dir, tmp_path, stem=primary.stem,
                         log=logs.append) is None
    assert any(f"{primary.stem}A0*" in m for m in logs)


def test_convert_final_defaults_to_primary_case_stem(tmp_path):
    """Omitting *stem* falls back to the primary (first) load case's stem."""
    cfg = Config.from_dict({
        "d3plot": {"enabled": True},
        "load_cases": [{"name": "pull", "stem": "deck_pull",
                        "sigma_allow": 300.0, "d_allow": 1.0}],
    })
    solve_dir = tmp_path / "case_0"
    solve_dir.mkdir()
    logs: list[str] = []
    assert convert_final(cfg, solve_dir, tmp_path, log=logs.append) is None
    assert any("deck_pullA0*" in m for m in logs)


def test_resolve_python_finds_posix_venv(tmp_path):
    """A provisioned POSIX venv (.venv/bin/python) was silently skipped --
    only the Windows Scripts/python.exe layout was probed, so on Linux the
    converter ran under sys.executable, which by design lacks its deps."""
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    assert _resolve_python(D3plotOpts(tool_root=str(tmp_path))) == str(py)

"""The isolated render-subprocess helper shared by report + animate.

Hermetic: the "render" scripts are trivial Python one-liners (no pyvista/GL), so
these exercise the launch / exit-code / timeout classification without a real
off-screen render.
"""
from __future__ import annotations

from oropt._render import RenderResult, run_render


def test_run_render_ok_on_clean_exit(tmp_path):
    out = tmp_path / "marker.txt"
    script = "import sys; open(sys.argv[1], 'w').write('ok')"
    result = run_render(script, [out], timeout_s=30)
    assert result.ok and result.returncode == 0
    assert out.read_text() == "ok"


def test_run_render_reports_nonzero_exit_with_last_line():
    script = "import sys; sys.stderr.write('boom: GL context failed\\n'); sys.exit(3)"
    result = run_render(script, [], timeout_s=30)
    assert not result.ok
    assert result.returncode == 3
    assert "boom: GL context failed" in result.detail
    assert not result.timed_out


def test_run_render_times_out():
    script = "import time; time.sleep(5)"
    result = run_render(script, [], timeout_s=0.5)
    assert not result.ok
    assert result.returncode is None
    assert result.timed_out


def test_render_result_timed_out_flag_distinguishes_launch_failure():
    # a None returncode that is NOT a timeout (e.g. launch failure) is not timed_out
    assert RenderResult(False, None, "could not launch renderer: x").timed_out is False
    assert RenderResult(False, None, "timed out after 1s").timed_out is True

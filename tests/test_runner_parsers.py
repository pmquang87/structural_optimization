"""Golden-fixture tests for the OpenRadioss output-listing parsers (E1/P3).

``oropt.runner`` scrapes the solver's free-text ``.out`` listings with a handful
of regexes to decide whether a starter/engine run succeeded, how many cycles it
ran, and whether an implicit solve is diverging. That seam has no other test
coverage and is exactly what an OpenRadioss version bump silently breaks, so the
snippets below are modelled on real listing format (0/N ``ERROR(S)`` summary,
``NORMAL``/``ERROR TERMINATION``, the ``--ITERATION DIVERGE--`` /
``--NEXT TIMESTEP IS DECREASED BY--`` retry cycle, the ``TOTAL NUMBER OF
CYCLES`` / ``ELAPSED TIME`` footer) and pinned as a regression guard.

Every assertion is derived from the actual logic in ``runner.py`` (the exact
return tuples/booleans/reason strings), and the headline property is checked in
both directions: a successful run is never read as a failure and a failed run is
never read as a success. Hermetic -- no OpenRadioss, no subprocess: the parsers
read a file, so each fixture is written to a tmp ``.out`` and parsed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from oropt.runner import (
    DivergenceMonitor,
    _engine_ok,
    _parse_engine_stats,
    _starter_ok,
)

# --------------------------------------------------------------------------- #
# Realistic listing fixtures (format observed on OpenRadioss / elevator-linkage
# runs). Kept faithful so an OR-format change flips a parser here first.
# --------------------------------------------------------------------------- #

STARTER_OK = """\
 ROOT: linkage    RESTART: 0000

 ..
 CHECK OF THE INPUT DATA
 ..
 MEMORY USED FOR I/O                     :        12 MB

 ------------------------------------------------------------
                          0 ERROR(S)
                          5 WARNING(S)
 ------------------------------------------------------------
"""

STARTER_ERRORS = """\
 ROOT: linkage    RESTART: 0000

 ..
 ** ERROR IN SOLID ELEMENT DEFINITION
 ID=60000123 HAS A NEGATIVE OR NULL VOLUME
 ..
 ------------------------------------------------------------
                          3 ERROR(S)
                          1 WARNING(S)
 ------------------------------------------------------------
"""

# A run cut off before the ERROR(S) summary ever printed (crash / kill / disk).
STARTER_TRUNCATED = """\
 ROOT: linkage    RESTART: 0000

 ..
 CHECK OF THE INPUT DATA
 READING NODES
"""

ENGINE_OK = """\
 ..
        CYCLE    TIME    TIME-STEP  ELEMENT          ERROR ...
         1  0.0000      0.1000E-01 SOLID   60000001   0.0%
       842  1.0000      0.1000E-01 SOLID   60000042   0.0%
 ..
 ANIMATION FILE: linkageA002 WRITTEN
 ..
 CUMULATIVE CPU TIME SUMMARY

 TOTAL NUMBER OF CYCLES  :         842

 ELAPSED TIME       =        1234.56 s
                    =  0 h 20 mn 34 s

 NORMAL TERMINATION
"""

ENGINE_ERROR_TERM = """\
 ..
        CYCLE    TIME    TIME-STEP  ELEMENT
         1  0.0000      0.1000E-01 SOLID   60000001
 ..
 ** ERROR : NEGATIVE VOLUME IN SOLID ELEMENT
 ERROR TERMINATION
"""

ENGINE_ERROR_ID = """\
 ..
        CYCLE    TIME    TIME-STEP
         1  0.0000      0.1000E-01
 ** ERROR ID :   305
 DESCRIPTION : negative jacobian
"""

# Diverging implicit solve, killed by the watchdog -> the listing is truncated
# mid-streak and NEVER reaches NORMAL TERMINATION (nor an ERROR marker).
ENGINE_DIVERGE_TRUNCATED = """\
 ..
   --ITERATION DIVERGE with MAX_ITER REACHED--

   --RESET ITERATION WITH NEW TIMESTEP--

     --NEXT TIMESTEP IS DECREASED BY-- 0.6667E+00
   --ITERATION DIVERGE with MAX_ITER REACHED--

   --RESET ITERATION WITH NEW TIMESTEP--

     --NEXT TIMESTEP IS DECREASED BY-- 0.6667E+00
"""

# Pure garbage / unrelated content (a wrong file, a corrupt tail).
ENGINE_GARBAGE = " lorem ipsum\n\x00\x00 not a solver listing at all\n"

# ---- diverge/recover listing atoms (real elevator-linkage format) -----------
DIVERGE_CYCLE = (
    "   --ITERATION DIVERGE with MAX_ITER REACHED--\n"
    " \n"
    "   --RESET ITERATION WITH NEW TIMESTEP--\n"
    " \n"
    "     --NEXT TIMESTEP IS DECREASED BY-- 0.6667E+00\n")
CONV_ROW = "         2              4.929E-01  4.653E-02  1.835E-02     C\n"
NONCONV_ROW = "        16              1.067E+00  5.917E-02  1.561E-02\n"
DT_INCREASE = "     --NEXT TIMESTEP IS INCREASED BY-- 0.1100E+01\n"


def _outfile(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# _starter_ok
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, exp_ok, exp_sub", [
    (STARTER_OK, True, "0 ERROR(S)"),
    (STARTER_ERRORS, False, "3 starter ERROR(S)"),
    (STARTER_TRUNCATED, False, "without an ERROR(S) summary"),
])
def test_starter_ok(tmp_path, text, exp_ok, exp_sub):
    ok, msg = _starter_ok(_outfile(tmp_path, "s_0000.out", text))
    assert ok is exp_ok
    assert exp_sub in msg


def test_starter_missing_file(tmp_path):
    ok, msg = _starter_ok(tmp_path / "nope_0000.out")
    assert ok is False
    assert "missing starter listing" in msg and "nope_0000.out" in msg


def test_starter_success_not_misread_as_failure(tmp_path):
    """The headline invariant: a clean starter (0 ERROR(S)) reads True and a
    failed one (N>0 ERROR(S)) reads False -- never crossed."""
    assert _starter_ok(_outfile(tmp_path, "a.out", STARTER_OK))[0] is True
    assert _starter_ok(_outfile(tmp_path, "b.out", STARTER_ERRORS))[0] is False


def test_starter_reports_the_error_count(tmp_path):
    ok, msg = _starter_ok(_outfile(tmp_path, "s.out",
                                   STARTER_OK.replace("0 ERROR(S)", "7 ERROR(S)")))
    assert ok is False and msg == "7 starter ERROR(S)"


# --------------------------------------------------------------------------- #
# _engine_ok
# --------------------------------------------------------------------------- #
def test_engine_normal_termination(tmp_path):
    ok, msg = _engine_ok(_outfile(tmp_path, "e_0001.out", ENGINE_OK))
    assert ok is True and msg == "NORMAL TERMINATION"


def test_engine_error_termination_surfaces_the_line(tmp_path):
    ok, msg = _engine_ok(_outfile(tmp_path, "e_0001.out", ENGINE_ERROR_TERM))
    assert ok is False
    assert msg == "ERROR TERMINATION"          # the offending line, stripped


def test_engine_error_id_marker(tmp_path):
    ok, msg = _engine_ok(_outfile(tmp_path, "e_0001.out", ENGINE_ERROR_ID))
    assert ok is False
    assert msg == "** ERROR ID :   305"        # matched via the 'ERROR ID' marker


def test_engine_diverge_truncated_is_a_failure(tmp_path):
    ok, msg = _engine_ok(_outfile(tmp_path, "e_0001.out", ENGINE_DIVERGE_TRUNCATED))
    assert ok is False and msg == "no NORMAL TERMINATION found"


def test_engine_garbage_is_a_failure(tmp_path):
    ok, msg = _engine_ok(_outfile(tmp_path, "e_0001.out", ENGINE_GARBAGE))
    assert ok is False and msg == "no NORMAL TERMINATION found"


def test_engine_missing_file(tmp_path):
    ok, msg = _engine_ok(tmp_path / "missing_0001.out")
    assert ok is False and "missing engine listing" in msg


def test_engine_success_not_misread_and_failure_not_misread(tmp_path):
    """The headline invariant, both directions: NORMAL TERMINATION -> True; every
    non-normal listing (error/diverge/garbage) -> False."""
    assert _engine_ok(_outfile(tmp_path, "ok.out", ENGINE_OK))[0] is True
    for name, text in [("err", ENGINE_ERROR_TERM), ("div", ENGINE_DIVERGE_TRUNCATED),
                       ("gar", ENGINE_GARBAGE)]:
        assert _engine_ok(_outfile(tmp_path, f"{name}.out", text))[0] is False


def test_engine_normal_wins_even_if_a_stray_error_word_appears(tmp_path):
    """NORMAL TERMINATION is checked first, so a benign 'ERROR' column header in
    the cycle table never flips a converged run to failed."""
    ok, msg = _engine_ok(_outfile(tmp_path, "e.out", ENGINE_OK))
    assert ok is True and msg == "NORMAL TERMINATION"


# --------------------------------------------------------------------------- #
# _parse_engine_stats
# --------------------------------------------------------------------------- #
def test_parse_engine_stats_reads_cycles_and_elapsed(tmp_path):
    cycles, elapsed = _parse_engine_stats(_outfile(tmp_path, "e.out", ENGINE_OK))
    assert cycles == 842
    assert elapsed == pytest.approx(1234.56)


def test_parse_engine_stats_missing_file(tmp_path):
    assert _parse_engine_stats(tmp_path / "gone.out") == (None, None)


def test_parse_engine_stats_absent_footer(tmp_path):
    """A truncated listing with neither footer line yields (None, None) rather
    than a crash -- the loop then records unknown cycles/time."""
    cycles, elapsed = _parse_engine_stats(
        _outfile(tmp_path, "e.out", ENGINE_DIVERGE_TRUNCATED))
    assert cycles is None and elapsed is None


def test_parse_engine_stats_cycles_only(tmp_path):
    text = " TOTAL NUMBER OF CYCLES  :         17\n (no elapsed line)\n"
    assert _parse_engine_stats(_outfile(tmp_path, "e.out", text)) == (17, None)


# --------------------------------------------------------------------------- #
# DivergenceMonitor -- fed incrementally the way runner._run_engine does
# --------------------------------------------------------------------------- #
def test_monitor_trips_only_after_the_configured_streak(tmp_path):
    """An unbroken run of diverge cycles trips exactly on the max_cycles-th one,
    not before -- fed chunk-by-chunk like the live listing poller."""
    mon = DivergenceMonitor(max_cycles=5)
    reasons = [mon.feed(DIVERGE_CYCLE) for _ in range(4)]
    assert all(r is None for r in reasons)     # 4 < 5: still alive
    reason = mon.feed(DIVERGE_CYCLE)           # the 5th consecutive
    assert reason is not None
    assert "5 consecutive" in reason and "diverge_max_cycles=5" in reason


def test_monitor_recovered_diverges_never_trip():
    """Isolated diverges that each recover (an accepted 'C' row + a timestep
    increase) reset the streak, so the healthy pattern never trips however long
    the run -- no false positive on a normal solve."""
    mon = DivergenceMonitor(max_cycles=5)
    healthy = DIVERGE_CYCLE + CONV_ROW + DT_INCREASE
    for _ in range(50):
        assert mon.feed(healthy) is None
    assert mon.streak == 0


def test_monitor_converged_row_resets_the_streak():
    mon = DivergenceMonitor(max_cycles=5)
    assert mon.feed(DIVERGE_CYCLE * 4) is None      # streak 4
    assert mon.feed(CONV_ROW) is None               # accepted step -> reset
    assert mon.feed(DIVERGE_CYCLE * 4) is None       # 4 again, no trip
    assert mon.feed(DIVERGE_CYCLE) is not None       # now the 5th consecutive


def test_monitor_nonconverged_rows_do_not_reset():
    """Retry noise between diverges (cycle/non-converged rows) is not an accepted
    step, so the streak keeps climbing to the trip point."""
    mon = DivergenceMonitor(max_cycles=5)
    stuck = DIVERGE_CYCLE + NONCONV_ROW
    reason = None
    for _ in range(5):
        reason = mon.feed(stuck)
    assert reason is not None and "5 consecutive" in reason


def test_monitor_is_chunking_invariant():
    """Byte-sized (mid-line) chunks trip at the same streak as one big feed --
    the exact way _read_new hands partial listing bytes to feed()."""
    text = DIVERGE_CYCLE * 5
    mon = DivergenceMonitor(max_cycles=5)
    tripped = [r for r in (mon.feed(text[i:i + 7])
                           for i in range(0, len(text), 7)) if r]
    assert tripped and "5 consecutive" in tripped[0]


def test_monitor_disabled_with_nonpositive_max_cycles():
    for mx in (0, -1):
        mon = DivergenceMonitor(max_cycles=mx)
        assert mon.feed(DIVERGE_CYCLE * 100) is None


def test_monitor_successful_listing_never_trips(tmp_path):
    """The whole successful engine listing, streamed through the monitor, never
    reports divergence."""
    mon = DivergenceMonitor(max_cycles=5)
    assert mon.feed(ENGINE_OK) is None
    assert mon.streak == 0

"""Per-iteration archiving and solve-dir isolation (logic only, no real solver)."""
from __future__ import annotations

from pathlib import Path

from oropt.loop import _archive_iteration, _clean_solve_dir


def _fake_solve_outputs(solve_dir: Path, stem: str) -> None:
    """Drop the files a finished OpenRadioss solve leaves behind in solve_dir."""
    solve_dir.mkdir(parents=True, exist_ok=True)
    (solve_dir / f"{stem}_0000.rad").write_text("mutated starter\n", encoding="utf-8")
    (solve_dir / f"{stem}_0001.rad").write_text("engine deck\n", encoding="utf-8")
    (solve_dir / f"{stem}_0000.out").write_text("0 ERROR(S)\n", encoding="utf-8")
    (solve_dir / f"{stem}_0001.out").write_text("NORMAL TERMINATION\n", encoding="utf-8")
    (solve_dir / f"{stem}A001").write_bytes(b"anim-state-1")
    (solve_dir / f"{stem}A002").write_bytes(b"anim-state-2")
    (solve_dir / f"{stem}_engine.log").write_text("engine log\n", encoding="utf-8")
    # the ~345 MB restart we must NOT copy (stand-in content)
    (solve_dir / f"{stem}_0000_0001.rst").write_bytes(b"x" * 8192)


def test_archive_iteration_copies_key_outputs(tmp_path):
    stem = "demo"
    solve_dir = tmp_path / "solve"
    work = tmp_path
    _fake_solve_outputs(solve_dir, stem)

    dest = _archive_iteration(solve_dir, work, stem, it=3)

    assert dest == work / "iter_0003"
    names = {p.name for p in dest.iterdir()}
    # mutated deck + engine listing + final animation state(s) only
    assert names == {f"{stem}_0000.rad", f"{stem}_0001.out",
                     f"{stem}A001", f"{stem}A002"}
    # the bulky restart and the irrelevant files are excluded
    assert not (dest / f"{stem}_0000_0001.rst").exists()
    assert not (dest / f"{stem}_0001.rad").exists()
    assert not (dest / f"{stem}_engine.log").exists()
    # content is preserved verbatim
    assert (dest / f"{stem}_0000.rad").read_text(encoding="utf-8") == "mutated starter\n"
    assert (dest / f"{stem}A002").read_bytes() == b"anim-state-2"


def test_archive_iteration_tolerates_missing_files(tmp_path):
    """A failed/partial solve leaves only some outputs — archive what exists."""
    stem = "demo"
    solve_dir = tmp_path / "solve"
    solve_dir.mkdir()
    (solve_dir / f"{stem}_0001.out").write_text("partial\n", encoding="utf-8")

    dest = _archive_iteration(solve_dir, tmp_path, stem, it=0)

    assert dest == tmp_path / "iter_0000"
    assert {p.name for p in dest.iterdir()} == {f"{stem}_0001.out"}


def test_clean_solve_dir_never_clobbers_source_deck(tmp_path):
    """When the run folder == the input case_dir, the mutated deck lives in
    case_dir/solve/ and wiping solve/ each iteration must leave the source
    case_dir/<stem>_0000.rad untouched."""
    stem = "demo"
    case_dir = tmp_path                       # run_folder == input folder
    source = case_dir / f"{stem}_0000.rad"
    source.write_text("PRISTINE SOURCE\n", encoding="utf-8")

    solve_dir = case_dir / "solve"
    _clean_solve_dir(solve_dir)
    mutated = solve_dir / f"{stem}_0000.rad"
    mutated.write_text("mutated for solve\n", encoding="utf-8")

    assert mutated.resolve() != source.resolve()      # genuinely different paths
    _clean_solve_dir(solve_dir)                        # next-iteration wipe
    assert source.read_text(encoding="utf-8") == "PRISTINE SOURCE\n"
    assert not mutated.exists()                        # only solve/ was wiped
    assert solve_dir.is_dir() and not any(solve_dir.iterdir())

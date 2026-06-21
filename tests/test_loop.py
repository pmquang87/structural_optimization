"""Per-iteration archiving and solve-dir isolation (logic only, no real solver)."""
from __future__ import annotations

from pathlib import Path

from oropt.config import Config
from oropt.loop import _archive_iteration, _clean_solve_dir

# A realistic per-load-case stem (the kind that lives on a load case while
# model.stem is left blank in a multi-load-case config).
MULTILOAD_STEM = "implicit_elevator-linkage_pull"


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


def test_archive_iteration_keeps_restart_when_requested(tmp_path):
    """With keep_restart the bulky restart is preserved alongside the curated
    outputs, giving a full per-iteration solver snapshot."""
    stem = "demo"
    solve_dir = tmp_path / "solve"
    _fake_solve_outputs(solve_dir, stem)

    dest = _archive_iteration(solve_dir, tmp_path, stem, it=2, keep_restart=True)

    rst = dest / f"{stem}_0000_0001.rst"
    assert rst.exists() and rst.read_bytes() == b"x" * 8192
    # the curated outputs are still archived too
    assert (dest / f"{stem}_0000.rad").exists()
    assert (dest / f"{stem}A002").exists()


def test_archive_iteration_tolerates_missing_files(tmp_path):
    """A failed/partial solve leaves only some outputs — archive what exists."""
    stem = "demo"
    solve_dir = tmp_path / "solve"
    solve_dir.mkdir()
    (solve_dir / f"{stem}_0001.out").write_text("partial\n", encoding="utf-8")

    dest = _archive_iteration(solve_dir, tmp_path, stem, it=0)

    assert dest == tmp_path / "iter_0000"
    assert {p.name for p in dest.iterdir()} == {f"{stem}_0001.out"}


def test_archive_iteration_multiload_stem_copies_all_outputs(tmp_path):
    """With a realistic per-case stem and keep_restart, the deck, listing, ALL
    animation states AND the restart are archived (the multi-load files live in
    a per-case ``solve/case_0`` dir)."""
    solve_dir = tmp_path / "solve" / "case_0"
    _fake_solve_outputs(solve_dir, MULTILOAD_STEM)

    dest = _archive_iteration(solve_dir, tmp_path, MULTILOAD_STEM, it=0,
                              keep_restart=True)

    assert {p.name for p in dest.iterdir()} == {
        f"{MULTILOAD_STEM}_0000.rad", f"{MULTILOAD_STEM}_0001.out",
        f"{MULTILOAD_STEM}A001", f"{MULTILOAD_STEM}A002",
        f"{MULTILOAD_STEM}_0000_0001.rst"}


def test_archive_iteration_subdir_nests_outputs_by_stem(tmp_path):
    """With *subdir* (the multi-load-case path) the curated outputs land in a
    stem-named sub-folder of the iteration folder, keeping each case's files
    grouped instead of side by side."""
    solve_dir = tmp_path / "solve" / "case_0"
    _fake_solve_outputs(solve_dir, MULTILOAD_STEM)

    dest = _archive_iteration(solve_dir, tmp_path, MULTILOAD_STEM, it=5,
                              keep_restart=True, subdir=MULTILOAD_STEM)

    assert dest == tmp_path / "iter_0005" / MULTILOAD_STEM
    assert {p.name for p in dest.iterdir()} == {
        f"{MULTILOAD_STEM}_0000.rad", f"{MULTILOAD_STEM}_0001.out",
        f"{MULTILOAD_STEM}A001", f"{MULTILOAD_STEM}A002",
        f"{MULTILOAD_STEM}_0000_0001.rst"}
    # the iteration folder itself holds only the per-case sub-folder
    assert {p.name for p in (tmp_path / "iter_0005").iterdir()} == {MULTILOAD_STEM}


def test_archive_uses_primary_case_stem_when_model_stem_blank(tmp_path):
    """Regression for the empty-stem trap: in a multi-load config ``model.stem``
    is blank and the real stems live per load case. The fixed call site archives
    by the PRIMARY case's stem (deck + listing + anim + rst); archiving by the
    old ``cfg.model.stem`` would have matched only the ``*.rst`` files."""
    cfg = Config.from_dict({
        "model": {"stem": ""},
        "load_cases": [
            {"name": "pull", "stem": MULTILOAD_STEM, "weight": 1.0},
            {"name": "push", "stem": "implicit_elevator-linkage_push", "weight": 1.0},
        ],
    })
    primary = cfg.load_case_list()[0]
    assert cfg.model.stem == ""               # the trap: model stem is blank ...
    assert primary.stem == MULTILOAD_STEM     # ... but the real stem is per-case

    solve_dir = tmp_path / "solve" / "case_0"
    _fake_solve_outputs(solve_dir, primary.stem)

    # Fixed call site: passes primary.stem -> the full curated snapshot lands.
    good = _archive_iteration(solve_dir, tmp_path / "good", primary.stem, it=0,
                              keep_restart=True)
    assert {p.name for p in good.iterdir()} == {
        f"{primary.stem}_0000.rad", f"{primary.stem}_0001.out",
        f"{primary.stem}A001", f"{primary.stem}A002",
        f"{primary.stem}_0000_0001.rst"}

    # Old buggy call site: cfg.model.stem == '' makes the .rad/.out/A0* patterns
    # match nothing while the `<stem>*.rst` glob degrades to `*.rst` and grabs
    # every restart -> only the .rst would be archived.
    bad = _archive_iteration(solve_dir, tmp_path / "bad", cfg.model.stem, it=0,
                             keep_restart=True)
    assert {p.name for p in bad.iterdir()} == {f"{primary.stem}_0000_0001.rst"}


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

"""Per-iteration archiving and solve-dir isolation (logic only, no real solver)."""
from __future__ import annotations

from pathlib import Path

import oropt.loop as loop
from oropt.config import Config
from oropt.loop import (_archive_iteration, _clean_solve_dir, _sequential_contract,
                        _solve_cases, copy_iter0, iter0_archive_dir,
                        resume_warnings, reuse_iter0_solve, snapshot_config_used)
from oropt.runner import RunResult
from oropt.status import Status

# A realistic per-load-case stem (the kind that lives on a load case).
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


def test_archive_uses_primary_case_stem(tmp_path):
    """The archive call site keys off the load case's own stem (deck + listing +
    anim + rst). Regression it guards: a blank stem would make the .rad/.out/A0*
    patterns match nothing while the ``<stem>*.rst`` glob degrades to ``*.rst`` and
    grabs every restart -> only the .rst would be archived."""
    cfg = Config.from_dict({
        "load_cases": [
            {"name": "pull", "stem": MULTILOAD_STEM, "weight": 1.0,
             "sigma_allow": 300.0, "d_allow": 1.0},
            {"name": "push", "stem": "implicit_elevator-linkage_push", "weight": 1.0,
             "sigma_allow": 300.0, "d_allow": 1.0},
        ],
    })
    primary = cfg.load_case_list()[0]
    assert primary.stem == MULTILOAD_STEM     # the real stem lives per-case

    solve_dir = tmp_path / "solve" / "case_0"
    _fake_solve_outputs(solve_dir, primary.stem)

    # Correct call site: passes primary.stem -> the full curated snapshot lands.
    good = _archive_iteration(solve_dir, tmp_path / "good", primary.stem, it=0,
                              keep_restart=True)
    assert {p.name for p in good.iterdir()} == {
        f"{primary.stem}_0000.rad", f"{primary.stem}_0001.out",
        f"{primary.stem}A001", f"{primary.stem}A002",
        f"{primary.stem}_0000_0001.rst"}

    # A blank stem makes the .rad/.out/A0* patterns match nothing while the
    # `<stem>*.rst` glob degrades to `*.rst` and grabs every restart -> only the
    # .rst would be archived.
    bad = _archive_iteration(solve_dir, tmp_path / "bad", "", it=0,
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


# ---- optimiser-switching provenance & guards -------------------------------
def test_snapshot_config_used_fresh_has_no_prior(tmp_path):
    cfg = Config(); cfg.optimizer = "beso"
    assert snapshot_config_used(tmp_path, cfg, resume=False) is None
    assert (tmp_path / "config_used.yaml").is_file()


def test_snapshot_config_used_resume_preserves_prior_stage(tmp_path):
    stage1 = Config(); stage1.optimizer = "beso"
    snapshot_config_used(tmp_path, stage1, resume=False)          # stage 1
    stage2 = Config(); stage2.optimizer = "levelset"
    prior = snapshot_config_used(tmp_path, stage2, resume=True)   # stage 2 (switch)
    assert prior == "beso"                                        # prior stage reported
    snaps = list(tmp_path.glob("config_used.2*.yaml"))           # timestamped preserve
    assert snaps, "prior stage config was not preserved"
    assert Config.from_yaml(snaps[0]).optimizer_name() == "beso"
    assert Config.from_yaml(tmp_path / "config_used.yaml").optimizer_name() == "levelset"


class _OC:                                    # minimal opts stub for resume_warnings
    def __init__(self, target=0.4, fr=1.0, pl=2):
        self.target_volume_fraction = target
        self.filter_radius = fr
        self.protect_layers = pl


def test_resume_warnings_flags_optimiser_switch():
    msgs = resume_warnings("beso", "levelset", cur_vf=0.45, oc=_OC(target=0.4))
    assert any("OPTIMISER SWITCHED beso -> levelset" in m for m in msgs)


def test_resume_warnings_flags_big_volume_gap_without_switch():
    msgs = resume_warnings("levelset", "levelset", cur_vf=0.70, oc=_OC(target=0.4))
    assert any("drive the volume down" in m for m in msgs)
    assert not any("SWITCHED" in m for m in msgs)                # same optimiser


def test_resume_warnings_quiet_when_aligned():
    # same optimiser, current volume within 0.1 of the target -> nothing to flag
    assert resume_warnings("beso", "beso", cur_vf=0.70, oc=_OC(target=0.68)) == []


# ---- iteration-0 solve reuse (copied iter_0000) ----------------------------
def test_iter0_archive_dir_single_vs_multi(tmp_path):
    assert iter0_archive_dir(tmp_path, "gp", 1) == tmp_path / "iter_0000"
    assert iter0_archive_dir(tmp_path, "gp", 2) == tmp_path / "iter_0000" / "gp"


def _seed_iter0(reuse_dir: Path, starter_text: str) -> None:
    reuse_dir.mkdir(parents=True, exist_ok=True)
    (reuse_dir / "gp_0000.rad").write_text(starter_text, encoding="utf-8")
    (reuse_dir / "gpA001").write_bytes(b"anim-final-state")
    (reuse_dir / "gp_0001.out").write_text("NORMAL TERMINATION\n", encoding="utf-8")


def test_reuse_iter0_matching_starter_copies_anim(tmp_path):
    reuse, solve = tmp_path / "iter_0000", tmp_path / "solve"
    solve.mkdir()
    _seed_iter0(reuse, "STARTER-DECK\n")
    starter = solve / "gp_0000.rad"
    starter.write_text("STARTER-DECK\n", encoding="utf-8")     # byte-identical
    res = reuse_iter0_solve(reuse, solve, "gp", starter, log=lambda *_: None)
    assert res is not None and res.ok                          # reuse accepted
    assert (solve / "gpA001").is_file()                        # animation copied in
    assert (solve / "gp_0001.out").is_file()


def test_reuse_iter0_starter_mismatch_solves_fresh(tmp_path):
    reuse, solve = tmp_path / "iter_0000", tmp_path / "solve"
    solve.mkdir()
    _seed_iter0(reuse, "STARTER-DECK-OLD\n")
    starter = solve / "gp_0000.rad"
    starter.write_text("STARTER-DECK-NEW\n", encoding="utf-8")  # different design
    logs: list[str] = []
    assert reuse_iter0_solve(reuse, solve, "gp", starter, log=logs.append) is None
    assert not (solve / "gpA001").exists()                     # nothing copied
    assert any("starter differs" in m for m in logs)


def test_reuse_iter0_missing_animation_solves_fresh(tmp_path):
    reuse, solve = tmp_path / "iter_0000", tmp_path / "solve"
    solve.mkdir()
    reuse.mkdir()
    (reuse / "gp_0000.rad").write_text("STARTER\n", encoding="utf-8")   # no anim
    starter = solve / "gp_0000.rad"
    starter.write_text("STARTER\n", encoding="utf-8")
    logs: list[str] = []
    assert reuse_iter0_solve(reuse, solve, "gp", starter, log=logs.append) is None
    assert any("no reusable" in m for m in logs)


# ---- copy_iter0 (GUI seed tool) --------------------------------------------
def test_copy_iter0_copies_tree(tmp_path):
    src, dst = tmp_path / "old", tmp_path / "new"
    (src / "iter_0000" / "pull").mkdir(parents=True)       # multi-case layout
    (src / "iter_0000" / "pull" / "pullA001").write_bytes(b"anim")
    dst.mkdir()
    ok, msg = copy_iter0(src, dst)
    assert ok and (dst / "iter_0000" / "pull" / "pullA001").is_file()


def test_copy_iter0_refuses_missing_source(tmp_path):
    ok, msg = copy_iter0(tmp_path / "old", tmp_path / "new")
    assert not ok and "no iter_0000" in msg


def test_copy_iter0_refuses_existing_without_overwrite_then_overwrites(tmp_path):
    src, dst = tmp_path / "old", tmp_path / "new"
    (src / "iter_0000").mkdir(parents=True)
    (src / "iter_0000" / "gpA001").write_bytes(b"new-anim")
    (dst / "iter_0000").mkdir(parents=True)
    (dst / "iter_0000" / "stale").write_text("old", encoding="utf-8")
    ok, msg = copy_iter0(src, dst)                          # exists, no overwrite
    assert not ok and "already has an iter_0000" in msg
    ok, _ = copy_iter0(src, dst, overwrite=True)            # overwrite replaces
    assert ok and (dst / "iter_0000" / "gpA001").is_file()
    assert not (dst / "iter_0000" / "stale").exists()       # stale content gone


def test_copy_iter0_refuses_same_folder(tmp_path):
    (tmp_path / "iter_0000").mkdir()
    ok, msg = copy_iter0(tmp_path, tmp_path)
    assert not ok and "same" in msg


# ---- concurrent per-iteration solves ---------------------------------------
def test_sequential_contract_stops_at_first_failure():
    ok = RunResult(ok=True, stage="ok", message="")
    bad = RunResult(ok=False, stage="engine", message="boom")
    # case 1 failed -> case_results is the [case 0] prefix, run_results ends at the
    # failure so run_results[-1] is it and cases[len(case_results)] == the failure.
    run_results, case_results = _sequential_contract(
        [(ok, "r0"), (bad, None), (ok, "r2")])
    assert len(case_results) == 1 and case_results == ["r0"]
    assert len(run_results) == 2 and run_results[-1] is bad


def _fake_cases(n):
    from types import SimpleNamespace
    return [SimpleNamespace(name=f"c{i}", stem=f"c{i}", fast_mode=False,
                            disp_constraints=[]) for i in range(n)]


def test_solve_cases_concurrent_runs_all_in_order(tmp_path, monkeypatch):
    cfg = Config(); cfg.run.solver_concurrency = 3
    cases = _fake_cases(3)
    calls: list[int] = []

    def fake_solve(cfg, case, cdeck, alive, no_pin, solve_dir, anim_dt,
                   exclude, fast_tie=None, reuse_dir=None, log=print):
        calls.append(int(case.stem[1:]))
        return RunResult(ok=True, stage="ok", message=""), f"res-{case.stem}"
    monkeypatch.setattr(loop, "_solve_case", fake_solve)

    run_results, case_results = _solve_cases(
        cfg, cases, [None] * 3, alive=None, no_pin=set(), solve_root=tmp_path,
        n_cases=3, exclude_elem_ids=None, fast_ties=[None] * 3,
        reuse_dirs=[None] * 3, status=Status(), work=tmp_path, it=0, log=lambda *_: None)
    assert sorted(calls) == [0, 1, 2]                       # every case solved
    assert case_results == ["res-c0", "res-c1", "res-c2"]   # collected in order


def test_solve_cases_concurrent_failure_truncates(tmp_path, monkeypatch):
    cfg = Config(); cfg.run.solver_concurrency = 3
    cases = _fake_cases(3)

    def fake_solve(cfg, case, cdeck, alive, no_pin, solve_dir, anim_dt,
                   exclude, fast_tie=None, reuse_dir=None, log=print):
        if case.stem == "c1":
            return RunResult(ok=False, stage="engine", message="boom"), None
        return RunResult(ok=True, stage="ok", message=""), f"res-{case.stem}"
    monkeypatch.setattr(loop, "_solve_case", fake_solve)

    run_results, case_results = _solve_cases(
        cfg, cases, [None] * 3, alive=None, no_pin=set(), solve_root=tmp_path,
        n_cases=3, exclude_elem_ids=None, fast_ties=[None] * 3,
        reuse_dirs=[None] * 3, status=Status(), work=tmp_path, it=0, log=lambda *_: None)
    assert case_results == ["res-c0"]                       # prefix before failure
    assert not run_results[-1].ok                           # ends at the failure


# --- d3plot animation source = last feasible iteration's archive ------------ #
def test_archived_iter_dir_layout(tmp_path):
    # Mirrors _archive_iteration: single case -> iter_NNNN/, multi -> iter_NNNN/<stem>/
    assert loop._archived_iter_dir(tmp_path, 5, "s", 1) == tmp_path / "iter_0005"
    assert (loop._archived_iter_dir(tmp_path, 28, MULTILOAD_STEM, 2)
            == tmp_path / "iter_0028" / MULTILOAD_STEM)


def test_final_anim_dir_prefers_archived_feasible_single_case(tmp_path):
    solve_dir = tmp_path / "solve"
    _fake_solve_outputs(solve_dir, MULTILOAD_STEM)
    _archive_iteration(solve_dir, tmp_path, MULTILOAD_STEM, 3)   # single-case archive
    got, used = loop._final_anim_dir(tmp_path, 3, MULTILOAD_STEM, 1, solve_dir)
    assert used is True and got == tmp_path / "iter_0003"
    assert sorted(got.glob(MULTILOAD_STEM + "A0*"))             # the archived anim


def test_final_anim_dir_prefers_archived_feasible_multi_case(tmp_path):
    solve_dir = tmp_path / "solve" / "case_0"
    _fake_solve_outputs(solve_dir, MULTILOAD_STEM)
    _archive_iteration(solve_dir, tmp_path, MULTILOAD_STEM, 7, subdir=MULTILOAD_STEM)
    got, used = loop._final_anim_dir(tmp_path, 7, MULTILOAD_STEM, 2, solve_dir)
    assert used is True and got == tmp_path / "iter_0007" / MULTILOAD_STEM


def test_final_anim_dir_falls_back_without_archive(tmp_path):
    # No iter_0003 archive -> convert the live (last-solved) solve dir instead.
    solve_dir = tmp_path / "solve"
    _fake_solve_outputs(solve_dir, MULTILOAD_STEM)
    got, used = loop._final_anim_dir(tmp_path, 3, MULTILOAD_STEM, 1, solve_dir)
    assert used is False and got == solve_dir


def test_final_anim_dir_falls_back_when_no_feasible_iteration(tmp_path):
    # feas_it == -1 (no feasible iteration) -> the fallback solve dir, even though
    # an archive happens to exist for other iterations.
    solve_dir = tmp_path / "solve"
    _fake_solve_outputs(solve_dir, MULTILOAD_STEM)
    _archive_iteration(solve_dir, tmp_path, MULTILOAD_STEM, 3)
    got, used = loop._final_anim_dir(tmp_path, -1, MULTILOAD_STEM, 1, solve_dir)
    assert used is False and got == solve_dir

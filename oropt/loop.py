"""The BESO optimisation loop: solve -> extract -> rank -> delete -> repeat.

Runs headless and writes status/history/topology every iteration so the GUI can
monitor without ever touching the run. Resumable from ``checkpoint.npz``.
"""
from __future__ import annotations

import dataclasses
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import status as st
from .beso import Beso, combine_sensitivity
from .config import Config, ResolvedCase
from .d3plot import convert_final
from .deck import Deck, prepare_engine
from .levelset import LevelSet
from .manufacturing import apply_manufacturing, manufacturing_active
from .animate import make_animation
from .mesh import Mesh
from .report import write_report
from .results import extract
from .runner import run_solver
from .smoothing import smooth_all_iterations, smooth_final
from .tobs import Tobs


def build_optimizer(cfg: Config, mesh: Mesh, protected: np.ndarray,
                    anchor: np.ndarray | None = None):
    """Construct the optimiser selected by ``cfg.optimizer``.

    All optimisers share the same interface (``volume_fraction``,
    ``raw_sensitivity``, ``filter_history``, ``next_target_vf``, ``update``,
    ``V0``), so the loop drives whichever is returned identically.
    """
    name = cfg.optimizer_name()
    if name == "levelset":
        return LevelSet(mesh, cfg.levelset, protected, anchor=anchor)
    if name == "tobs":
        return Tobs(mesh, cfg.tobs, protected, anchor=anchor)
    if name == "beso":
        return Beso(mesh, cfg.beso, protected, anchor=anchor)
    raise ValueError(
        f"unknown optimizer {cfg.optimizer!r} (expected 'beso', 'levelset' or 'tobs')")


def collect_protect_nodes(deck: Deck, model, include_bc: bool = True) -> np.ndarray:
    """Seed nodes whose elements must be frozen: the BC/symmetry set (included
    unless *include_bc* is False) plus any user-defined keep-out regions
    (``freeze_group_ids`` /GRNOD/NODE groups, e.g. 99999999, and explicit
    ``freeze_node_ids``)."""
    parts = []
    if include_bc:
        parts.append(deck.group_nodes(model.bc_group_id))
    for gid in getattr(model, "freeze_group_ids", []) or []:
        parts.append(deck.group_nodes(int(gid)))
    explicit = getattr(model, "freeze_node_ids", []) or []
    if explicit:
        parts.append(np.asarray([int(v) for v in explicit], dtype=np.int64))
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)


def _clean_solve_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)


def _case_solve_dir(solve_root: Path, n_cases: int, i: int) -> Path:
    """Solve directory for load case *i*. A single case uses ``solve/`` directly
    (so a classic run is byte-identical); multiple cases each get their own
    ``solve/case_<i>/`` so their decks, listings and animations never collide."""
    return solve_root if n_cases == 1 else solve_root / f"case_{i}"


def _case_config(cfg: Config, case: ResolvedCase) -> Config:
    """A shallow copy of *cfg* whose ``model.stem`` / ``model.disp_node_id`` are
    the load case's, so the existing ``run_solver`` / ``extract`` (which key off
    the model stem and disp node) operate on that case with no other change."""
    return dataclasses.replace(
        cfg, model=dataclasses.replace(cfg.model, stem=case.stem,
                                       disp_node_id=case.disp_node_id))


def _solve_case(cfg: Config, case: ResolvedCase, deck: Deck, alive: np.ndarray,
                no_pin: set, solve_dir: Path, anim_dt: float):
    """Write the alive deck for one load case, solve it, extract its results.

    The per-case "solve + extract" unit reused for every case each iteration.
    Returns ``(run_result, results)`` where *results* is ``None`` if the solve
    failed (the caller surfaces *run_result*)."""
    case_cfg = _case_config(cfg, case)
    _clean_solve_dir(solve_dir)
    deck.write(solve_dir / f"{case.stem}_0000.rad", alive, no_pin=no_pin)
    prepare_engine(case.engine, solve_dir / f"{case.stem}_0001.rad", anim_dt=anim_dt)
    res = run_solver(case_cfg, solve_dir)
    if not res.ok:
        return res, None
    return res, extract(case_cfg, solve_dir)


def _archive_iteration(solve_dir: Path, work: Path, stem: str, it: int,
                       keep_restart: bool = False,
                       subdir: str | None = None) -> Path:
    """Copy iteration *it*'s key OpenRadioss outputs into ``work/iter_{it:04d}/``.

    Preserves the small, replay-worthy artefacts before ``solve_dir`` is wiped for
    the next iteration: the mutated starter deck (``<stem>_0000.rad``), the engine
    listing (``<stem>_0001.out``) and the final animation state(s) (``<stem>A0*``).
    The ~345 MB restart (``<stem>*.rst``) is skipped unless *keep_restart* is set,
    in which case the full solver state is kept too. Missing files are skipped
    (e.g. after a failed solve).

    *subdir* nests the outputs one level deeper (``iter_{it:04d}/<subdir>/``) so a
    multi-load-case run keeps each case's files in its own stem-named folder
    instead of side by side; left ``None`` (the single-case path) the files land
    directly in the iteration folder, byte-identical to a classic run."""
    dest = work / f"iter_{it:04d}"
    if subdir:
        dest = dest / subdir
    dest.mkdir(parents=True, exist_ok=True)
    for name in (f"{stem}_0000.rad", f"{stem}_0001.out"):
        src = solve_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    for anim in sorted(solve_dir.glob(f"{stem}A0*")):
        if anim.is_file():
            shutil.copy2(anim, dest / anim.name)
    if keep_restart:
        for rst in sorted(solve_dir.glob(f"{stem}*.rst")):
            if rst.is_file():
                shutil.copy2(rst, dest / rst.name)
    return dest


def _converged(vfs: list[float], feasible: bool, target_vf: float,
               window: int, tol: float) -> bool:
    if not feasible or len(vfs) < window:
        return False
    if vfs[-1] > target_vf * (1.0 + 1e-6):
        return False
    recent = vfs[-window:]
    return (max(recent) - min(recent)) / max(np.mean(recent), 1e-12) < tol


def run_optimization(cfg: Config, resume: bool = False,
                     log: Callable[[str], None] = print,
                     should_stop: Optional[Callable[[], bool]] = None) -> st.Status:
    work = cfg.work()
    solve_root = work / "solve"
    m = cfg.model

    # Snapshot the exact config this run uses into its own run folder (which is the
    # case_dir when work_dir is blank, or the queue's per-run --work-dir folder), so
    # every result set carries the config that produced it. Best-effort: a write
    # failure is logged, never fatal.
    try:
        cfg.to_yaml(work / "config_used.yaml")
    except Exception as exc:  # noqa: BLE001
        log(f"[oropt] could not write config_used.yaml: {exc}")

    (work / "stop.flag").unlink(missing_ok=True)   # ignore any stale stop request
    if should_stop is None:                          # GUI "Stop" drops a stop.flag
        def should_stop() -> bool:
            return (work / "stop.flag").exists()

    # Run-level knobs shared by both optimisers (protect_*, archive_*, max_iter,
    # convergence_*, target_volume_fraction) come from the selected optimiser's
    # config block, keeping the loop optimiser-agnostic.
    oc = cfg.active_opts()

    # ---- load cases --------------------------------------------------------
    # A single (default) case == the classic single-solve run. The primary case's
    # deck defines the shared geometry/mesh/protected set; every other case must
    # share the same design-part element ids (only its load cards differ).
    cases = cfg.load_case_list()
    n_cases = len(cases)
    primary = cases[0]

    log(f"[oropt] loading deck {primary.starter}"
        + (f" (+{n_cases - 1} more load case(s))" if n_cases > 1 else ""))
    deck = Deck.load(primary.starter, m.design_part_id, m.design_node_min)
    case_decks = [deck]
    for case in cases[1:]:
        cdeck = Deck.load(case.starter, m.design_part_id, m.design_node_min)
        if not np.array_equal(cdeck.elem_ids, deck.elem_ids):
            raise ValueError(
                f"load case {case.name!r} (stem {case.stem!r}) has a different "
                f"design-part element set than the primary case {primary.name!r}; "
                "all load cases must share the same mesh")
        case_decks.append(cdeck)
    mesh = Mesh.from_deck(deck)
    bc_nodes = deck.group_nodes(m.bc_group_id)
    no_pin = set(int(v) for v in bc_nodes)            # already kinematically constrained
    protect_bc = getattr(oc, "protect_bc_nodes", True)
    frozen_nodes = collect_protect_nodes(deck, m, include_bc=protect_bc)  # BC frozen unless opted out
    log(f"[oropt] {deck.n_design_elements} design elements; "
        f"{bc_nodes.size} BC nodes ({'frozen' if protect_bc else 'deletable'}); "
        f"{frozen_nodes.size} frozen seed nodes; building protected set + filter ...")
    protected = mesh.protected_mask(deck, frozen_nodes,
                                    contact_dist=oc.contact_protect_dist,
                                    layers=oc.protect_layers)
    # The BC/load region always anchors connectivity (so floating islands are
    # still dropped sensibly) even when its elements are allowed to be deleted.
    if protect_bc:
        anchor = protected
    else:
        anchor_nodes = collect_protect_nodes(deck, m, include_bc=True)
        anchor = mesh.protected_mask(deck, anchor_nodes,
                                     contact_dist=oc.contact_protect_dist,
                                     layers=oc.protect_layers)
    opt = build_optimizer(cfg, mesh, protected, anchor=anchor)
    log(f"[oropt] optimizer={cfg.optimizer_name()}; protected elements: "
        f"{int(protected.sum())} ({100*protected.mean():.1f}%); V0={opt.V0:.3f}")

    # ---- initial / resumed state ------------------------------------------
    alive = np.ones(deck.n_design_elements, dtype=bool)
    sens_prev: Optional[np.ndarray] = None
    start_iter = 0
    if resume:
        ckpt = st.load_checkpoint(work)
        if ckpt is not None:
            alive = ckpt["alive_mask"]; sens_prev = ckpt["sens_prev"]
            start_iter = ckpt["iteration"]
            log(f"[oropt] resumed at iteration {start_iter}, "
                f"vf={opt.volume_fraction(alive):.3f}")

    pid = st.write_pid(work)
    status = st.Status(state="running", max_iter=oc.max_iter,
                       elements_total=deck.n_design_elements,
                       sigma_allow=cfg.constraints.sigma_allow,
                       d_allow=cfg.constraints.d_allow, pid=pid)
    st.write_status(work, status)

    vfs: list[float] = []
    elapsed = 0.0
    try:
        for it in range(start_iter, oc.max_iter):
            if should_stop and should_stop():
                status.state = "stopped"; status.message = "stop requested"
                break

            log(f"[oropt] iter {it}: vf={opt.volume_fraction(alive):.3f} "
                f"alive={int(alive.sum())} -> solving "
                + (f"{n_cases} load cases ..." if n_cases > 1 else "..."))
            # ---- solve every load case (sequentially, each in its own dir) -
            t0 = time.time()
            run_results: list = []
            case_results = []
            for i, (case, cdeck) in enumerate(zip(cases, case_decks)):
                csolve = _case_solve_dir(solve_root, n_cases, i)
                res, r = _solve_case(cfg, case, cdeck, alive, no_pin,
                                     csolve, cfg.run.anim_dt)
                run_results.append(res)
                if r is None:                       # this case failed -> abort iter
                    break
                case_results.append(r)
            iter_wall = time.time() - t0
            elapsed += iter_wall

            res = run_results[-1]
            if not res.ok:
                failed = cases[len(case_results)]
                status.state = "failed"
                status.message = (f"{res.stage}: {res.message}" if n_cases == 1
                                  else f"case {failed.name!r}: {res.stage}: {res.message}")
                status.iteration = it
                status.or_termination = res.message
                log(f"[oropt] SOLVE FAILED: {status.message}")
                break

            # ---- combine cases: weighted-sum sensitivity + worst-case gate -
            raws = [opt.raw_sensitivity(r, deck.elem_ids, alive)
                    for r in case_results]
            raw = combine_sensitivity(raws, [c.weight for c in cases])
            sigma_max = max(r.sigma_max for r in case_results)   # worst over cases
            disp = max(r.disp for r in case_results)             # worst over cases
            feasible = all(r.sigma_max <= case.sigma_allow
                           and r.disp <= case.d_allow
                           for case, r in zip(cases, case_results))
            vf = opt.volume_fraction(alive)
            vfs.append(vf)

            # ---- publish state for the GUI --------------------------------
            remaining = oc.max_iter - it - 1
            status = st.Status(
                state="running", iteration=it, max_iter=oc.max_iter,
                volume_fraction=vf, sigma_max=sigma_max,
                sigma_allow=cfg.constraints.sigma_allow, disp=disp,
                d_allow=cfg.constraints.d_allow, feasible=feasible,
                elements_alive=int(alive.sum()),
                elements_total=deck.n_design_elements,
                or_termination=res.message, iter_wall_s=iter_wall,
                elapsed_s=elapsed, eta_s=iter_wall * remaining,
                message=("feasible" if feasible else "INFEASIBLE - backing off"),
                pid=pid)
            st.write_status(work, status)
            st.append_history(work, {
                "iteration": it, "volume_fraction": round(vf, 6),
                "sigma_max": round(sigma_max, 4), "disp": round(disp, 6),
                "elements_alive": int(alive.sum()), "feasible": feasible,
                "iter_wall_s": round(iter_wall, 1), "or_termination": res.message})
            sens = opt.filter_history(raw, sens_prev)
            st.write_topology(work, deck.node_xyz, mesh.conn_rows, alive,
                              fields={"sensitivity": sens,
                                      "vonmises": _scatter_max(case_results,
                                                               deck.elem_ids)},
                              iteration=it)
            if oc.archive_iterations:
                # Archive EVERY load case's curated outputs (mutated deck +
                # listing + animation state(s), plus the restart when
                # archive_restart) into work/iter_NNNN/. With multiple cases each
                # case's files go into their own stem-named sub-folder
                # (iter_NNNN/<stem>/) so a case's deck/listing/anim/restart stay
                # grouped instead of intermixed; a single-case run archives
                # straight into iter_NNNN/, byte-identical to before.
                # Archive by each case's own stem (== model.stem for a classic
                # single-case run, but the real per-case stem when model.stem is
                # blank in a multi-load-case config) so the deck/listing/anim are
                # matched, not just the restart files.
                for i, case in enumerate(cases):
                    _archive_iteration(_case_solve_dir(solve_root, n_cases, i),
                                       work, case.stem, it,
                                       keep_restart=oc.archive_restart,
                                       subdir=case.stem if n_cases > 1 else None)
            log(f"[oropt] iter {it}: sigma_max={sigma_max:.2f}/"
                f"{cfg.constraints.sigma_allow} disp={disp:.4f}/"
                f"{cfg.constraints.d_allow} feasible={feasible} "
                f"({iter_wall:.0f}s)")

            # ---- convergence ----------------------------------------------
            if _converged(vfs, feasible, oc.target_volume_fraction,
                          oc.convergence_window, oc.convergence_tol):
                status.state = "converged"
                status.message = "converged at target volume, feasible"
                st.write_status(work, status)
                log("[oropt] CONVERGED")
                break

            # ---- next design ----------------------------------------------
            sens_prev = sens
            target_vf = opt.next_target_vf(vf, feasible)
            alive = opt.update(alive, sens, target_vf)
            # Additive-manufacturing printability constraints (min member size,
            # symmetry, overhang) on the fresh alive mask; re-drop islands a
            # constraint may have created. No-op unless configured.
            if manufacturing_active(cfg.manufacturing):
                alive = apply_manufacturing(alive, mesh, cfg.manufacturing, protected)
                alive = mesh.keep_connected(alive, anchor)
            st.save_checkpoint(work, it + 1, alive, sens_prev)
        else:
            status.state = "converged" if status.state == "running" else status.state
            status.message = status.message or "reached max_iter"
    finally:
        if status.state == "running":
            status.state = "stopped"
        st.write_status(work, status)
        # Post-run: best-effort OpenRadioss anim -> LS-Dyna d3plot of the final
        # design, for EVERY load case (each in its own solve dir). Done while the
        # run still owns the pid (so the GUI stays 'running' and won't recycle
        # solve/ mid-conversion); never let post-processing affect the run's
        # result/state.
        for i, case in enumerate(cases):
            try:
                # Use a per-case cfg so convert_final keys off that case's stem
                # (== model.stem for a single-case run, but the real stem when
                # model.stem is blank in a multi-load-case config) and finds its
                # <stem>A0* animation rather than nothing. Distinct stems -> the
                # per-case d3plot files never collide in work/d3plot/.
                convert_final(_case_config(cfg, case),
                              _case_solve_dir(solve_root, n_cases, i), work, log)
            except Exception as exc:  # noqa: BLE001
                log(f"[oropt] d3plot: unexpected error during conversion: {exc}")
        try:
            smooth_final(cfg, work, log)
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] smooth: unexpected error during smoothing: {exc}")
        # Smooth every per-iteration snapshot too (topology_smoothed_iterNNNN.<ext>)
        # so the smoothed shape evolution is reviewable, not just the final design.
        try:
            smooth_all_iterations(cfg, work, log)
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] smooth: unexpected error during per-iteration smoothing: {exc}")
        # Automatic topology-evolution GIF from the per-iteration smoothed surfaces
        # (raw snapshots as fallback). Isolated off-screen render like the report's;
        # best-effort, never affects the run. Built *before* the report so the
        # report can embed it.
        try:
            make_animation(cfg, work, log)
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] animate: unexpected error during animation: {exc}")
        # Automatic post-run summary (report.html/report.md) from the status &
        # history this run wrote. Read-only and best-effort; never affects the run.
        try:
            write_report(cfg, work, log)
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] report: unexpected error during report: {exc}")
        st.clear_pid(work)
    return status


def _scatter(results, elem_ids: np.ndarray) -> np.ndarray:
    """von-Mises mapped onto the full element array (0 where no result)."""
    out = np.zeros(elem_ids.size, dtype=float)
    pos = np.searchsorted(elem_ids, results.element_ids)
    valid = (pos < elem_ids.size) & \
        (elem_ids[np.clip(pos, 0, elem_ids.size - 1)] == results.element_ids)
    out[pos[valid]] = results.vonmises[valid]
    return out


def _scatter_max(results_list, elem_ids: np.ndarray) -> np.ndarray:
    """Per-element worst (max) von-Mises across load cases, on the full element
    array. With one case this is exactly :func:`_scatter` of that case."""
    out = _scatter(results_list[0], elem_ids)
    for results in results_list[1:]:
        out = np.maximum(out, _scatter(results, elem_ids))
    return out

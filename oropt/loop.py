"""The BESO optimisation loop: solve -> extract -> rank -> delete -> repeat.

Runs headless and writes status/history/topology every iteration so the GUI can
monitor without ever touching the run. Resumable from ``checkpoint.npz``.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import status as st
from .beso import Beso
from .config import Config
from .d3plot import convert_final
from .deck import Deck, prepare_engine
from .levelset import LevelSet
from .manufacturing import apply_manufacturing, manufacturing_active
from .mesh import Mesh
from .results import extract
from .runner import run_solver
from .smoothing import smooth_final


def build_optimizer(cfg: Config, mesh: Mesh, protected: np.ndarray,
                    anchor: np.ndarray | None = None):
    """Construct the optimiser selected by ``cfg.optimizer``.

    Both optimisers share the same interface (``volume_fraction``,
    ``raw_sensitivity``, ``filter_history``, ``next_target_vf``, ``update``,
    ``V0``), so the loop drives whichever is returned identically.
    """
    name = cfg.optimizer_name()
    if name == "levelset":
        return LevelSet(mesh, cfg.levelset, protected, anchor=anchor)
    if name == "beso":
        return Beso(mesh, cfg.beso, protected, anchor=anchor)
    raise ValueError(
        f"unknown optimizer {cfg.optimizer!r} (expected 'beso' or 'levelset')")


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


def _archive_iteration(solve_dir: Path, work: Path, stem: str, it: int,
                       keep_restart: bool = False) -> Path:
    """Copy iteration *it*'s key OpenRadioss outputs into ``work/iter_{it:04d}/``.

    Preserves the small, replay-worthy artefacts before ``solve_dir`` is wiped for
    the next iteration: the mutated starter deck (``<stem>_0000.rad``), the engine
    listing (``<stem>_0001.out``) and the final animation state(s) (``<stem>A0*``).
    The ~345 MB restart (``<stem>*.rst``) is skipped unless *keep_restart* is set,
    in which case the full solver state is kept too. Missing files are skipped
    (e.g. after a failed solve)."""
    dest = work / f"iter_{it:04d}"
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
    solve_dir = work / "solve"
    stem = cfg.model.stem
    m = cfg.model

    (work / "stop.flag").unlink(missing_ok=True)   # ignore any stale stop request
    if should_stop is None:                          # GUI "Stop" drops a stop.flag
        should_stop = lambda: (work / "stop.flag").exists()

    # Run-level knobs shared by both optimisers (protect_*, archive_*, max_iter,
    # convergence_*, target_volume_fraction) come from the selected optimiser's
    # config block, keeping the loop optimiser-agnostic.
    oc = cfg.active_opts()

    log(f"[oropt] loading deck {m.starter()}")
    deck = Deck.load(m.starter(), m.design_part_id, m.design_node_min)
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

            _clean_solve_dir(solve_dir)
            deck.write(solve_dir / f"{stem}_0000.rad", alive, no_pin=no_pin)
            prepare_engine(m.engine(), solve_dir / f"{stem}_0001.rad",
                           anim_dt=cfg.run.anim_dt)

            log(f"[oropt] iter {it}: vf={opt.volume_fraction(alive):.3f} "
                f"alive={int(alive.sum())} -> solving ...")
            t0 = time.time()
            res = run_solver(cfg, solve_dir)
            iter_wall = time.time() - t0
            elapsed += iter_wall

            if not res.ok:
                status.state = "failed"
                status.message = f"{res.stage}: {res.message}"
                status.iteration = it
                status.or_termination = res.message
                log(f"[oropt] SOLVE FAILED: {status.message}")
                break

            r = extract(cfg, solve_dir)
            vf = opt.volume_fraction(alive)
            feasible = bool(r.sigma_max <= cfg.constraints.sigma_allow
                            and r.disp <= cfg.constraints.d_allow)
            vfs.append(vf)

            # ---- publish state for the GUI --------------------------------
            remaining = oc.max_iter - it - 1
            status = st.Status(
                state="running", iteration=it, max_iter=oc.max_iter,
                volume_fraction=vf, sigma_max=r.sigma_max,
                sigma_allow=cfg.constraints.sigma_allow, disp=r.disp,
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
                "sigma_max": round(r.sigma_max, 4), "disp": round(r.disp, 6),
                "elements_alive": int(alive.sum()), "feasible": feasible,
                "iter_wall_s": round(iter_wall, 1), "or_termination": res.message})
            raw = opt.raw_sensitivity(r, deck.elem_ids, alive)
            sens = opt.filter_history(raw, sens_prev)
            st.write_topology(work, deck.node_xyz, mesh.conn_rows, alive,
                              fields={"sensitivity": sens,
                                      "vonmises": _scatter(r, deck.elem_ids)},
                              iteration=it)
            if oc.archive_iterations:
                _archive_iteration(solve_dir, work, stem, it,
                                   keep_restart=oc.archive_restart)
            log(f"[oropt] iter {it}: sigma_max={r.sigma_max:.2f}/"
                f"{cfg.constraints.sigma_allow} disp={r.disp:.4f}/"
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
        # design. Done while the run still owns the pid (so the GUI stays
        # 'running' and won't recycle solve/ mid-conversion); never let
        # post-processing affect the run's result/state.
        try:
            convert_final(cfg, solve_dir, work, log)
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] d3plot: unexpected error during conversion: {exc}")
        try:
            smooth_final(cfg, work, log)
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] smooth: unexpected error during smoothing: {exc}")
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

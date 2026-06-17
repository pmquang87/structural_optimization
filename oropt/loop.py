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
from .deck import Deck, prepare_engine
from .mesh import Mesh
from .results import extract
from .runner import run_solver


def collect_protect_nodes(deck: Deck, model) -> np.ndarray:
    """Seed nodes whose elements must be frozen: the BC/symmetry set plus any
    user-defined keep-out regions (``freeze_group_ids`` /GRNOD/NODE groups, e.g.
    99999999, and explicit ``freeze_node_ids``)."""
    parts = [deck.group_nodes(model.bc_group_id)]
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


def _archive_iteration(solve_dir: Path, work: Path, stem: str, it: int) -> Path:
    """Copy iteration *it*'s key OpenRadioss outputs into ``work/iter_{it:04d}/``.

    Preserves the small, replay-worthy artefacts before ``solve_dir`` is wiped for
    the next iteration: the mutated starter deck (``<stem>_0000.rad``), the engine
    listing (``<stem>_0001.out``) and the final animation state(s) (``<stem>A0*``).
    The ~345 MB restart (``<stem>_0000_0001.rst``) is deliberately *not* copied.
    Missing files are skipped (e.g. after a failed solve)."""
    dest = work / f"iter_{it:04d}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in (f"{stem}_0000.rad", f"{stem}_0001.out"):
        src = solve_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    for anim in sorted(solve_dir.glob(f"{stem}A0*")):
        if anim.is_file():
            shutil.copy2(anim, dest / anim.name)
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

    log(f"[oropt] loading deck {m.starter()}")
    deck = Deck.load(m.starter(), m.design_part_id, m.design_node_min)
    mesh = Mesh.from_deck(deck)
    bc_nodes = deck.group_nodes(m.bc_group_id)
    no_pin = set(int(v) for v in bc_nodes)            # already kinematically constrained
    protect_nodes = collect_protect_nodes(deck, m)    # BC + user keep-out sets
    log(f"[oropt] {deck.n_design_elements} design elements; "
        f"{bc_nodes.size} BC nodes; {protect_nodes.size} protected seed nodes; "
        f"building protected set + filter ...")
    protected = mesh.protected_mask(deck, protect_nodes,
                                    contact_dist=cfg.beso.contact_protect_dist,
                                    layers=cfg.beso.protect_layers)
    beso = Beso(mesh, cfg.beso, protected)
    log(f"[oropt] protected elements: {int(protected.sum())} "
        f"({100*protected.mean():.1f}%); V0={beso.V0:.3f}")

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
                f"vf={beso.volume_fraction(alive):.3f}")

    pid = st.write_pid(work)
    status = st.Status(state="running", max_iter=cfg.beso.max_iter,
                       elements_total=deck.n_design_elements,
                       sigma_allow=cfg.constraints.sigma_allow,
                       d_allow=cfg.constraints.d_allow, pid=pid)
    st.write_status(work, status)

    vfs: list[float] = []
    elapsed = 0.0
    try:
        for it in range(start_iter, cfg.beso.max_iter):
            if should_stop and should_stop():
                status.state = "stopped"; status.message = "stop requested"
                break

            _clean_solve_dir(solve_dir)
            deck.write(solve_dir / f"{stem}_0000.rad", alive, no_pin=no_pin)
            prepare_engine(m.engine(), solve_dir / f"{stem}_0001.rad",
                           anim_dt=cfg.run.anim_dt)

            log(f"[oropt] iter {it}: vf={beso.volume_fraction(alive):.3f} "
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
            vf = beso.volume_fraction(alive)
            feasible = bool(r.sigma_max <= cfg.constraints.sigma_allow
                            and r.disp <= cfg.constraints.d_allow)
            vfs.append(vf)

            # ---- publish state for the GUI --------------------------------
            remaining = cfg.beso.max_iter - it - 1
            status = st.Status(
                state="running", iteration=it, max_iter=cfg.beso.max_iter,
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
            raw = beso.raw_sensitivity(r, deck.elem_ids, alive)
            sens = beso.filter_history(raw, sens_prev)
            st.write_topology(work, deck.node_xyz, mesh.conn_rows, alive,
                              fields={"sensitivity": sens,
                                      "vonmises": _scatter(r, deck.elem_ids)},
                              iteration=it)
            if cfg.beso.archive_iterations:
                _archive_iteration(solve_dir, work, stem, it)
            log(f"[oropt] iter {it}: sigma_max={r.sigma_max:.2f}/"
                f"{cfg.constraints.sigma_allow} disp={r.disp:.4f}/"
                f"{cfg.constraints.d_allow} feasible={feasible} "
                f"({iter_wall:.0f}s)")

            # ---- convergence ----------------------------------------------
            if _converged(vfs, feasible, cfg.beso.target_volume_fraction,
                          cfg.beso.convergence_window, cfg.beso.convergence_tol):
                status.state = "converged"
                status.message = "converged at target volume, feasible"
                st.write_status(work, status)
                log("[oropt] CONVERGED")
                break

            # ---- next design ----------------------------------------------
            sens_prev = sens
            target_vf = beso.next_target_vf(vf, feasible)
            alive = beso.update(alive, sens, target_vf)
            st.save_checkpoint(work, it + 1, alive, sens_prev)
        else:
            status.state = "converged" if status.state == "running" else status.state
            status.message = status.message or "reached max_iter"
    finally:
        if status.state == "running":
            status.state = "stopped"
        st.write_status(work, status)
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

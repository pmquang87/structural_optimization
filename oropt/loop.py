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
from .hca import Hca
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
    if name == "hca":
        return Hca(mesh, cfg.hca, protected, anchor=anchor)
    if name == "beso":
        return Beso(mesh, cfg.beso, protected, anchor=anchor)
    raise ValueError(
        f"unknown optimizer {cfg.optimizer!r} "
        "(expected 'beso', 'levelset', 'tobs' or 'hca')")


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


def collect_stress_exclude_nodes(deck: Deck, model) -> np.ndarray:
    """Nodes whose design elements have their von-Mises ignored: the user-defined
    stress-exclusion /GRNOD/NODE groups (``stress_exclude_group_ids``, e.g.
    999999998) plus explicit ``stress_exclude_node_ids``.

    Empty by default, so a config that doesn't use the feature is unaffected. The
    elements touching these nodes still take part in the optimisation (they are not
    frozen); only their stress is dropped from ``sigma_max`` / feasibility / the
    Monitor & report (see :func:`oropt.results.parse_vtk`)."""
    parts = []
    for gid in getattr(model, "stress_exclude_group_ids", []) or []:
        parts.append(deck.group_nodes(int(gid)))
    explicit = getattr(model, "stress_exclude_node_ids", []) or []
    if explicit:
        parts.append(np.asarray([int(v) for v in explicit], dtype=np.int64))
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)


def stress_exclude_mask(deck: Deck, mesh: Mesh, model) -> np.ndarray:
    """Boolean mask (aligned with ``deck.elem_ids``) of design elements whose
    von-Mises is ignored — those touching a stress-exclusion node. All-False when
    the feature is unconfigured."""
    nodes = collect_stress_exclude_nodes(deck, model)
    if not nodes.size:
        return np.zeros(deck.n_design_elements, dtype=bool)
    # layers=0 / contact_dist=0 -> exactly the elements touching an excluded node.
    return mesh.protected_mask(deck, nodes, contact_dist=0.0, layers=0)


def resolve_growth_boxes(deck: Deck, boxes) -> list:
    """Return *boxes* with every ``deck_box_id`` reference resolved to concrete
    geometry read from the starter deck's ``/BOX/{RECTA,SPHER,CYLIN}`` cards.

    A growth region may name a ``/BOX/...`` card authored in the pre-processor
    (``deck_box_id``) instead of literal coordinates; here that region's ``shape``
    and coordinates (and, for a ``/BOX/RECTA`` with a ``/SKEW/FIX`` skew, its local
    frame) are filled from :meth:`oropt.deck.Deck.box`, so everything downstream —
    selection, guards, the overlay — treats it exactly like a coordinate region.
    Regions without a ``deck_box_id`` are returned unchanged. Raises ``ValueError``
    when the referenced card is absent from the deck."""
    out = []
    for i, b in enumerate(boxes or []):
        if getattr(b, "deck_box_id", None) is None:
            out.append(b)
            continue
        spec = deck.box(b.deck_box_id)
        label = b.name or f"#{i + 1}"
        if spec is None:
            raise ValueError(
                f"growth box {label!r} references box id {b.deck_box_id} but no "
                "/BOX/RECTA, /BOX/SPHER or /BOX/CYLIN card with that id is in the "
                "deck; author it in the pre-processor or give literal coordinates")
        out.append(dataclasses.replace(b, deck_box_id=None, **spec))
    return out


def growth_candidate_mask(deck: Deck, mesh: Mesh, model,
                          log: Callable[[str], None] = print) -> np.ndarray:
    """Boolean mask (aligned with ``deck.elem_ids``) of growth-candidate
    elements: those whose centroid lies inside one of ``model.growth_boxes``.
    Candidates start the run *void* and may be grown into by the optimiser's
    bi-directional update. All-False when no boxes are configured.

    Raises ``ValueError`` at run start — before any ~13-min solve — for the
    setup mistakes that would otherwise waste (or silently no-op) a multi-hour
    run:

    * a box selecting **no** design elements: the box volume was not pre-meshed
      into the design part;
    * candidate connectivity referencing node ids below ``design_node_min``: the
      free-node guard could not pin those nodes while the elements are void, so
      the implicit tangent would go singular;
    * candidates **unreachable** from the initial structure (no shared-node path,
      even through other candidates): a non-conformal interface —
      ``keep_connected`` would drop anything grown there as a floating island.
    """
    boxes = resolve_growth_boxes(deck, getattr(model, "growth_boxes", []) or [])
    if not boxes:
        return np.zeros(deck.n_design_elements, dtype=bool)

    candidate = np.zeros(deck.n_design_elements, dtype=bool)
    box_masks = []
    for i, b in enumerate(boxes):
        bm = mesh.in_boxes_mask([b])
        label = b.name or f"#{i + 1}"
        if not bm.any():
            raise ValueError(
                f"growth box {label!r} contains no design elements -- the region "
                "volume must be pre-meshed into the design part "
                f"(/TETRA4/{deck.design_part_id}) before material can grow "
                "there, or generated with the growth-mesh step "
                "(python -m oropt.growthmesh / the GUI's Generate button)")
        log(f"[oropt] growth box {label!r}: {int(bm.sum())} candidate elements "
            "start void")
        box_masks.append((label, bm))
        candidate |= bm

    # Void-element nodes must be pinnable by the free-node guard, which only
    # covers design nodes (ids >= design_node_min).
    cand_nodes = deck.elem_conn[candidate]
    bad = cand_nodes < deck.design_node_min
    if bad.any():
        raise ValueError(
            f"growth-box candidate elements reference "
            f"{int(np.unique(cand_nodes[bad]).size)} node id(s) below "
            f"design_node_min={deck.design_node_min}; the free-node guard cannot "
            "pin them while the elements are void (singular implicit tangent). "
            "Renumber the expansion-mesh nodes to ids >= design_node_min")

    # Every candidate must be connected (via shared nodes, possibly through
    # other candidates) to the initially-alive structure, or it can never be
    # grown: keep_connected would drop it as a floating island every iteration.
    alive0 = ~candidate
    reachable = mesh.keep_connected(np.ones_like(candidate), alive0)
    unreachable = candidate & ~reachable
    if unreachable.any():
        names = [label for label, bm in box_masks if (bm & unreachable).any()]
        raise ValueError(
            f"{int(unreachable.sum())} growth-box candidate element(s) in "
            f"box(es) {', '.join(repr(n) for n in names)} share no nodes with "
            "the initial structure (directly or through other candidates), so "
            "they could never be grown. Make the expansion mesh node-conformal "
            "with the part (imprint the part surface, then merge/equivalence "
            "the coincident interface nodes)")
    return candidate


@dataclasses.dataclass
class BoxPreview:
    """One growth region's preview row: how many design elements it would void."""
    name: str
    shape: str
    count: int          # design elements whose centroid lies inside the region
    note: str = ""      # a per-region issue (empty region / unresolved deck box)


@dataclasses.dataclass
class GrowthPreview:
    """Result of previewing a config's growth regions against a loaded deck."""
    rows: list          # list[BoxPreview], one per configured region
    total_candidates: int   # unique elements across all regions (0 if a guard trips)
    total_elements: int     # deck.n_design_elements
    guard: str = ""         # run-start guard error that would abort a run, if any


def preview_growth_boxes(deck: Deck, mesh: Mesh, model) -> GrowthPreview:
    """Count the design elements each growth region would start *void*, without
    launching a run — the data behind the GUI's "preview regions" button.

    Per region: the centroid-in-region element count (0 flags a region whose
    volume was not pre-meshed into the design part), plus a note for a region
    referencing a ``/BOX/RECTA`` card absent from the deck. Also runs the real
    run-start guards (:func:`growth_candidate_mask`) and reports, in ``guard``, the
    first error that would abort a run (an empty region, candidate nodes below
    ``design_node_min``, or an unreachable candidate); ``guard`` is ``""`` when the
    regions are run-ready, with ``total_candidates`` the unique element count that
    would then start void. Never raises."""
    boxes = getattr(model, "growth_boxes", []) or []
    rows: list = []
    for i, b in enumerate(boxes):
        label = b.name or f"#{i + 1}"
        try:
            resolved = resolve_growth_boxes(deck, [b])[0]
        except ValueError as exc:
            rows.append(BoxPreview(label, b.shape_kind(), 0, str(exc)))
            continue
        count = int(mesh.in_boxes_mask([resolved]).sum())
        note = ("" if count else "no design elements inside -- the region volume "
                "is not pre-meshed into the design part")
        rows.append(BoxPreview(label, resolved.shape_kind(), count, note))
    total, guard = 0, ""
    if boxes:
        try:
            total = int(growth_candidate_mask(
                deck, mesh, model, log=lambda _m: None).sum())
        except ValueError as exc:
            guard = str(exc)
    return GrowthPreview(rows=rows, total_candidates=total,
                         total_elements=deck.n_design_elements, guard=guard)


def _clean_solve_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)


def _case_solve_dir(solve_root: Path, n_cases: int, i: int) -> Path:
    """Solve directory for load case *i*. A single case uses ``solve/`` directly
    (so a classic run is byte-identical); multiple cases each get their own
    ``solve/case_<i>/`` so their decks, listings and animations never collide."""
    return solve_root if n_cases == 1 else solve_root / f"case_{i}"


def _within(value: float, limit: Optional[float]) -> bool:
    """True if *value* satisfies *limit*; a blank limit (``None``) is unconstrained."""
    return limit is None or value <= limit


def disp_breakdown(case, res) -> list[tuple]:
    """Per-node displacement rows for a case: ``(node_id, |disp|, d_allow,
    feasible)`` for each of the case's displacement constraints. The displacement
    is read from ``res.disps`` (``nan`` for a node absent from the animation, which
    fails any set limit)."""
    rows = []
    for dc in case.disp_constraints:
        val = float(res.disps.get(dc.node_id, float("nan")))
        rows.append((dc.node_id, val, dc.d_allow, _within(val, dc.d_allow)))
    return rows


def worst_disp(rows) -> tuple[float, Optional[float]]:
    """Headline ``(disp, d_allow)`` for a set of :func:`disp_breakdown` rows: the
    constraint with the worst utilisation ratio ``value/limit``. Unconstrained rows
    (blank limit) rank below any constrained one; among only-unconstrained rows the
    largest displacement is surfaced. ``(nan, None)`` when there are no rows."""
    if not rows:
        return float("nan"), None
    constrained = [(v, lim) for _, v, lim, _ in rows if lim is not None]
    if constrained:
        return max(constrained, key=lambda p: p[0] / p[1])
    finite = [(v, None) for _, v, _, _ in rows if v == v]
    if finite:
        return max(finite, key=lambda p: p[0])
    return float("nan"), None


def stress_ratio_field(cases, case_results, elem_ids: np.ndarray
                       ) -> Optional[np.ndarray]:
    """Per-element worst ``vonmises/sigma_allow`` across the stress-limited load
    cases, on the full element array (0 where no result — dead/absent elements).
    ``None`` when no case sets a stress limit. Feeds the stress-responsive
    add-back bias: elements above 1.0 are the overstressed region material
    should be recovered next to."""
    out = None
    for case, res in zip(cases, case_results):
        if case.sigma_allow is None:
            continue
        f = _scatter(res, elem_ids) / case.sigma_allow
        out = f if out is None else np.maximum(out, f)
    return out


def worst_violation(cases, case_results) -> float:
    """Worst constraint-utilisation ratio ``value/limit`` over all load cases and
    both limit types (``sigma_max/sigma_allow`` and, for *every* displacement
    constraint, ``disp/d_allow``). A blank limit (``None``) is unconstrained and
    skipped; with no limits at all the design is trivially feasible and 0.0 is
    returned. ``v <= 1`` <=> feasible — this is the violation *magnitude* the
    proportional back-off controller reacts to (see
    :func:`oropt.beso.gate_target_vf`), where the feasible flag only says on/off.
    """
    ratios = []
    for case, res in zip(cases, case_results):
        if case.sigma_allow is not None:
            ratios.append(res.sigma_max / case.sigma_allow)
        for _nid, val, limit, _feas in disp_breakdown(case, res):
            if limit is not None:
                ratios.append(val / limit)
    return max(ratios, default=0.0)


def _solve_case(cfg: Config, case: ResolvedCase, deck: Deck, alive: np.ndarray,
                no_pin: set, solve_dir: Path, anim_dt: float,
                exclude_elem_ids: Optional[np.ndarray] = None):
    """Write the alive deck for one load case, solve it, extract its results.

    The per-case "solve + extract" unit reused for every case each iteration. The
    case's ``stem`` / ``disp_node_id`` are passed straight to ``run_solver`` /
    ``extract`` (they accept them explicitly), so one *cfg* drives every case.
    *exclude_elem_ids* (the stress-exclusion set, shared by all cases) is forwarded
    to ``extract`` so the case's ``sigma_max`` ignores those elements. Returns
    ``(run_result, results)`` where *results* is ``None`` if the solve failed (the
    caller surfaces *run_result*)."""
    _clean_solve_dir(solve_dir)
    deck.write(solve_dir / f"{case.stem}_0000.rad", alive, no_pin=no_pin)
    prepare_engine(case.engine, solve_dir / f"{case.stem}_0001.rad", anim_dt=anim_dt)
    res = run_solver(cfg, solve_dir, stem=case.stem)
    if not res.ok:
        return res, None
    return res, extract(cfg, solve_dir, stem=case.stem,
                        disp_node_ids=[dc.node_id for dc in case.disp_constraints],
                        exclude_element_ids=exclude_elem_ids)


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
    # One load case == the classic single-solve run. The primary case's deck
    # defines the shared geometry/mesh/protected set; every other case must share
    # the same design-part element ids (only its load cards differ).
    cases = cfg.load_case_list()
    if not cases:
        raise ValueError("no load cases defined -- nothing to solve "
                         "(define at least one load case)")
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
    # Growth boxes: candidate elements start void and may be grown into. They are
    # never protected -- a candidate inside the protected dilation would otherwise
    # be force-materialised at iteration 1 regardless of sensitivity. (New array,
    # so the anchor mask above keeps its own view.)
    candidate = growth_candidate_mask(deck, mesh, m, log=log)
    n_candidate = int(candidate.sum())
    if n_candidate:
        overlap = int((protected & candidate).sum())
        if overlap:
            log(f"[oropt] growth: {overlap} candidate elements overlapped the "
                "protected set -> left unprotected (candidates are never frozen)")
        protected = protected & ~candidate
        log(f"[oropt] growth: {n_candidate} candidate elements "
            f"({100 * candidate.mean():.1f}% of the design space) start void; "
            "volume fractions are relative to the enlarged (part + boxes) space")
    opt = build_optimizer(cfg, mesh, protected, anchor=anchor)
    log(f"[oropt] optimizer={cfg.optimizer_name()}; protected elements: "
        f"{int(protected.sum())} ({100*protected.mean():.1f}%); V0={opt.V0:.3f}")

    # Stress-exclusion region: design elements touching a user-flagged hot-spot
    # node set have their von-Mises ignored everywhere it's reported (sigma_max,
    # feasibility, the Monitor & report stress field). Computed once — all load
    # cases share the design mesh. The elements still take part in the optimisation.
    stress_excluded = stress_exclude_mask(deck, mesh, m)
    exclude_elem_ids = deck.elem_ids[stress_excluded]
    n_excluded = int(exclude_elem_ids.size)
    if n_excluded:
        log(f"[oropt] stress-exclusion: {n_excluded} design elements "
            f"({100*stress_excluded.mean():.1f}%) ignored for sigma_max / "
            "feasibility / monitor / report")

    # ---- initial / resumed state ------------------------------------------
    alive = ~candidate            # everything alive except growth-box candidates
    sens_prev: Optional[np.ndarray] = None
    start_iter = 0
    if resume:
        ckpt = st.load_checkpoint(work)
        if ckpt is not None:
            if ckpt["alive_mask"].shape != alive.shape:
                raise ValueError(
                    f"checkpoint alive mask has {ckpt['alive_mask'].size} "
                    f"elements but the deck has {deck.n_design_elements} -- the "
                    "design mesh changed since this run was checkpointed (e.g. "
                    "growth boxes were pre-meshed in); start a fresh run instead "
                    "of resuming")
            alive = ckpt["alive_mask"]; sens_prev = ckpt["sens_prev"]
            start_iter = ckpt["iteration"]
            log(f"[oropt] resumed at iteration {start_iter}, "
                f"vf={opt.volume_fraction(alive):.3f}")

    pid = st.write_pid(work)
    # Placeholder headline limits until the first iteration publishes real ones:
    # the primary case's stress limit and its tightest displacement limit.
    primary_d_limits = [dc.d_allow for dc in primary.disp_constraints
                        if dc.d_allow is not None]
    status = st.Status(state="running", max_iter=oc.max_iter,
                       elements_total=deck.n_design_elements,
                       stress_excluded_elems=n_excluded,
                       elements_candidate=n_candidate,
                       sigma_allow=(primary.sigma_allow if primary.sigma_allow
                                    is not None else float("nan")),
                       d_allow=(min(primary_d_limits) if primary_d_limits
                                else float("nan")), pid=pid)
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
                                     csolve, cfg.run.anim_dt, exclude_elem_ids)
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
            # Each case is gated against its OWN limits; a blank limit (None) leaves
            # that quantity unconstrained. A case with several displacement
            # constraints is feasible only when EVERY one holds. A design is
            # feasible only when every case is. per_case carries the full
            # per-node breakdown for the GUI; each case's headline disp/d_allow is
            # its worst-ratio displacement constraint.
            disp_rows = [disp_breakdown(case, r)
                         for case, r in zip(cases, case_results)]
            per_case = []
            for case, r, rows in zip(cases, case_results, disp_rows):
                c_disp, c_d_allow = worst_disp(rows)
                disp_feasible = all(feas for *_, feas in rows)
                per_case.append({
                    "name": case.name,
                    "sigma_max": float(r.sigma_max), "sigma_allow": case.sigma_allow,
                    "disp": float(c_disp), "d_allow": c_d_allow,
                    "disp_constraints": [
                        {"node_id": nid, "disp": float(val), "d_allow": lim,
                         "feasible": bool(feas)}
                        for nid, val, lim, feas in rows],
                    "feasible": bool(_within(r.sigma_max, case.sigma_allow)
                                     and disp_feasible)})
            feasible = all(c["feasible"] for c in per_case)
            violation = worst_violation(cases, case_results)
            # Headline sigma_max stays the worst raw stress across cases; headline
            # disp is the worst *ratio* displacement constraint over every case and
            # every node. Each is reported with the limit of the constraint it came
            # from (a blank limit -> NaN, "no limit") so the Monitor's "limit"
            # matches what gated feasibility.
            si = max(range(n_cases), key=lambda i: case_results[i].sigma_max)
            sigma_max = case_results[si].sigma_max
            sigma_allow = cases[si].sigma_allow
            sigma_allow = float("nan") if sigma_allow is None else sigma_allow
            disp, d_allow = worst_disp([row for rows in disp_rows for row in rows])
            d_allow = float("nan") if d_allow is None else d_allow
            vf = opt.volume_fraction(alive)
            vfs.append(vf)

            # ---- publish state for the GUI --------------------------------
            remaining = oc.max_iter - it - 1
            status = st.Status(
                state="running", iteration=it, max_iter=oc.max_iter,
                volume_fraction=vf, sigma_max=sigma_max,
                sigma_allow=sigma_allow, disp=disp,
                d_allow=d_allow, feasible=feasible,
                elements_alive=int(alive.sum()),
                elements_total=deck.n_design_elements,
                stress_excluded_elems=n_excluded,
                elements_candidate=n_candidate,
                elements_grown=int((alive & candidate).sum()),
                cases=per_case,
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
            vm_field = _scatter_max(case_results, deck.elem_ids)
            if n_excluded:                       # drop the ignored stress from any view
                vm_field[stress_excluded] = np.nan
            st.write_topology(work, deck.node_xyz, mesh.conn_rows, alive,
                              fields={"sensitivity": sens, "vonmises": vm_field},
                              iteration=it)
            if oc.archive_iterations:
                # Archive EVERY load case's curated outputs (mutated deck +
                # listing + animation state(s), plus the restart when
                # archive_restart) into work/iter_NNNN/. With multiple cases each
                # case's files go into their own stem-named sub-folder
                # (iter_NNNN/<stem>/) so a case's deck/listing/anim/restart stay
                # grouped instead of intermixed; a single-case run archives
                # straight into iter_NNNN/, byte-identical to before.
                # Archive by each case's own stem so the deck/listing/anim are
                # matched, not just the restart files.
                for i, case in enumerate(cases):
                    _archive_iteration(_case_solve_dir(solve_root, n_cases, i),
                                       work, case.stem, it,
                                       keep_restart=oc.archive_restart,
                                       subdir=case.stem if n_cases > 1 else None)
            log(f"[oropt] iter {it}: sigma_max={sigma_max:.2f}/"
                f"{sigma_allow} disp={disp:.4f}/"
                f"{d_allow} feasible={feasible} "
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
            target_vf = opt.next_target_vf(vf, feasible, violation=violation)
            # Stress-responsive add-back bias (off by default): when a stress
            # limit is violated, scale the sensitivity driving THIS update by
            # (1 + bias * vonmises/sigma_allow) so the material the back-off
            # recovers lands near the overstressed region, not wherever the
            # energy ranking happens to point. The ratio field is spatially
            # filtered so the overstress bleeds into the neighbouring void
            # elements — the ones add-back can actually resurrect. Transient:
            # sens_prev / the published sensitivity stay unbiased.
            update_sens = sens
            stress_infeasible = any(
                c["sigma_allow"] is not None and c["sigma_max"] > c["sigma_allow"]
                for c in per_case)
            if oc.addback_stress_bias > 0.0 and stress_infeasible:
                ratio = stress_ratio_field(cases, case_results, deck.elem_ids)
                if n_excluded:               # excluded hot-spots attract nothing
                    ratio[stress_excluded] = 0.0
                ratio = opt.filter_history(ratio, None)
                update_sens = sens * (1.0 + oc.addback_stress_bias * ratio)
            alive = opt.update(alive, update_sens, target_vf)
            # Manufacturing constraints (min/max member size, symmetry, casting,
            # extrusion, overhang) on the fresh alive mask; re-drop islands a
            # constraint may have created. No-op unless configured. The unbiased
            # sensitivity guides the max-member carve toward the least-useful
            # material.
            if manufacturing_active(cfg.manufacturing):
                alive = apply_manufacturing(alive, mesh, cfg.manufacturing,
                                            protected, sensitivity=sens)
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
                # Pass the case's own stem so convert_final finds its <stem>A0*
                # animation. Distinct stems -> the per-case d3plot files never
                # collide in work/d3plot/.
                convert_final(cfg, _case_solve_dir(solve_root, n_cases, i),
                              work, stem=case.stem, log=log)
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

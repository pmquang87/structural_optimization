"""The BESO optimisation loop: solve -> extract -> rank -> delete -> repeat.

Runs headless and writes status/history/topology every iteration so the GUI can
monitor without ever touching the run. Resumable from ``checkpoint.npz``.
"""
from __future__ import annotations

import dataclasses
import difflib
import filecmp
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import status as st
from .beso import Beso, combine_sensitivity, id_positions
from .config import Config, ResolvedCase
from .d3plot import convert_final
from .deck import Deck, prepare_engine
from .fastmode import FastModeTie, build_fast_case, discover_tie
from .hca import Hca
from .keepout import resolve_deck_region, resolve_keepout
from .levelset import LevelSet
from .manufacturing import apply_manufacturing, manufacturing_active
from .animate import make_animation
from .mesh import Mesh
from .report import reported_iteration, write_report
from .results import extract
from .runner import RunResult, run_solver
from .controller import build_backoff_controller, build_weight_controller
from .sanity import audit as sanity_audit
from .mfg_verify import verify as mfg_verify
from .saip import Saip
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
    if name == "saip":
        return Saip(mesh, cfg.saip, protected, anchor=anchor)
    if name == "beso":
        return Beso(mesh, cfg.beso, protected, anchor=anchor)
    raise ValueError(
        f"unknown optimizer {cfg.optimizer!r} "
        "(expected 'beso', 'levelset', 'tobs', 'hca' or 'saip')")


def snapshot_config_used(work: Path, cfg: Config, resume: bool,
                         log: Callable[[str], None] = print) -> Optional[str]:
    """Write ``config_used.yaml`` for this run; return the *prior* stage's optimiser.

    Fresh run: just writes the snapshot. ``--resume``: preserves the existing
    ``config_used.yaml`` as ``config_used.<timestamp>.yaml`` before overwriting, so
    a run continued with changed parameters / a switched optimiser keeps every
    stage's config instead of only the last, and returns the prior stage's
    optimiser name (or ``None``) so the caller can flag a switch. Best-effort: a
    write failure is logged, never fatal."""
    prior_optimizer: Optional[str] = None
    used = work / "config_used.yaml"
    try:
        if resume and used.is_file():
            try:
                prior_optimizer = Config.from_yaml(used).optimizer_name()
            except Exception:  # noqa: BLE001
                prior_optimizer = None
            shutil.copy2(used, work / f"config_used.{time.strftime('%Y%m%d-%H%M%S')}.yaml")
        cfg.to_yaml(used)
    except Exception as exc:  # noqa: BLE001
        log(f"[oropt] could not write config_used.yaml: {exc}")
    return prior_optimizer


def resume_warnings(prior_optimizer: Optional[str], optimizer: str,
                    cur_vf: float, oc) -> list[str]:
    """The messages a resume should log about how it silently changes the run.

    Two footguns of continuing a stopped run with edits: (1) a *different*
    optimiser swaps the whole knob block and re-inits the continuous field from
    the alive mask; (2) a target well below the resumed volume means a big removal
    is coming (a switched-in block can carry a different ``target_volume_fraction``
    — e.g. beso 0.7 vs levelset 0.4). Pure function so it is unit-testable without
    a solve; the loop just logs whatever it returns."""
    msgs: list[str] = []
    if prior_optimizer and prior_optimizer != optimizer:
        msgs.append(
            f"[oropt] resume: OPTIMISER SWITCHED {prior_optimizer} -> {optimizer}; "
            "the alive mask carries over, the continuous field re-initialises from "
            f"it, and ALL run knobs now come from the {optimizer} block "
            f"(target_vf={oc.target_volume_fraction:g}, "
            f"filter_radius={oc.filter_radius:g}, protect_layers={oc.protect_layers})")
    if cur_vf - oc.target_volume_fraction > 0.1:
        msgs.append(
            f"[oropt] resume: current vf={cur_vf:.3f} is well above this stage's "
            f"target_volume_fraction={oc.target_volume_fraction:g}; it will drive "
            f"the volume down by ~{cur_vf - oc.target_volume_fraction:.2f} -- "
            "confirm that's intended (a switched-in block can carry a different target)")
    return msgs


# Consecutive "gate asked to grow, the design shrank anyway" iterations before
# the loop warns: 1-2 can be benign quantisation, 3 is a controller-defeating
# trend (the elevator-linkage run showed 6 before its web collapse).
GROW_STALL_ITERS = 3


def _grow_stall(count: int, prev_target_vf: Optional[float],
                prev_vf: Optional[float], vf: float) -> int:
    """Update the consecutive-iteration counter for 'the volume gate requested
    growth (target above the vf it was computed from) yet the achieved volume
    fell'. One such iteration is noise; a run of them means removal outside the
    volume controller's accounting is outrunning it."""
    if prev_target_vf is None or prev_vf is None:
        return 0
    if prev_target_vf > prev_vf + 1e-12 and vf < prev_vf - 1e-12:
        return count + 1
    return 0


def _removal_spike(removed: int, recent: list[int], factor: int = 10,
                   floor: int = 50) -> bool:
    """True when one update removed *factor*x more elements than any recent one
    (and more than *floor*, so quiet phases don't trip on noise). Catches a
    wholesale plateau/web collapse before a multi-hour solve is spent on the
    severed design. Needs >= 3 history entries, so a legitimately large first
    nucleation carve never fires it."""
    if len(recent) < 3:
        return False
    return removed > max(factor * max(recent), floor)


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


def validate_group_ids(deck: Deck, model) -> None:
    """Fail fast on configured /GRNOD/NODE group ids that select zero nodes.

    :meth:`oropt.deck.Deck.group_nodes` silently returns empty for an id with
    no matching ``/GRNOD/NODE/<id>`` block, so a typo'd ``bc_group_id`` /
    ``freeze_group_ids`` / ``stress_exclude_group_ids`` entry would quietly
    disable its region and let a multi-hour run be steered by results the
    config meant to alter (a stress-exclusion id one digit off once kept a
    load-introduction hot-spot in sigma_max, flagging a feasible design
    INFEASIBLE for the whole run). Raises ``ValueError`` at run start --
    before any ~13-min solve -- naming the offending setting and the deck's
    actual group ids, and distinguishing an id **absent** from the deck
    (likely a typo; the closest existing id is suggested) from a group that
    exists but **lists no nodes** (fix the group in the pre-processor)."""
    existing = deck.group_ids()
    have = set(existing)
    problems: list[str] = []
    checks = (("bc_group_id", [getattr(model, "bc_group_id", None)]),
              ("freeze_group_ids", getattr(model, "freeze_group_ids", []) or []),
              ("stress_exclude_group_ids",
               getattr(model, "stress_exclude_group_ids", []) or []))
    for setting, gids in checks:
        for gid in gids:
            if gid is None:
                continue
            gid = int(gid)
            if gid in have:
                if not deck.group_nodes(gid).size:
                    problems.append(f"{setting}: /GRNOD/NODE/{gid} exists in "
                                    "the deck but lists no nodes")
                continue
            close = difflib.get_close_matches(
                str(gid), [str(g) for g in existing], n=1)
            hint = f" (closest in the deck: {close[0]})" if close else ""
            problems.append(f"{setting}: no /GRNOD/NODE/{gid} in the deck{hint}")
    if problems:
        if existing:
            avail = ", ".join(str(g) for g in existing[:20])
            if len(existing) > 20:
                avail += f", ... ({len(existing)} total)"
        else:
            avail = "none"
        raise ValueError(
            "configured node group(s) select zero nodes -- "
            + "; ".join(problems)
            + f". /GRNOD/NODE groups in the deck: {avail}. A zero-node group "
            "silently disables its BC-protection / keep-out / stress-exclusion "
            "region, so the run stops now instead of hours in")


def stress_exclude_mask(deck: Deck, mesh: Mesh, model) -> np.ndarray:
    """Boolean mask (aligned with ``deck.elem_ids``) of design elements whose
    von-Mises is ignored — those touching a stress-exclusion node. All-False when
    the feature is unconfigured."""
    nodes = collect_stress_exclude_nodes(deck, model)
    if not nodes.size:
        return np.zeros(deck.n_design_elements, dtype=bool)
    # layers=0 / contact_dist=0 -> exactly the elements touching an excluded node.
    return mesh.protected_mask(deck, nodes, contact_dist=0.0, layers=0)


def active_growth_boxes(model) -> list:
    """The growth boxes that take effect at run time: the configured list, or
    none when growth is switched off (``model.growth_enabled`` False). The boxes
    stay on ``model`` either way so the GUI preserves them across a toggle — this
    is the single gate the run-time consumers (candidate / blocked masks) share,
    so a disabled config can never grow material regardless of leftover boxes."""
    if not getattr(model, "growth_enabled", True):
        return []
    return getattr(model, "growth_boxes", []) or []


def resolve_growth_boxes(deck: Deck, boxes, case_dir=None) -> list:
    """Return *boxes* with every deferred region reference resolved to concrete
    geometry: a ``deck_box_id`` filled from the starter deck's
    ``/BOX/{RECTA,SPHER,CYLIN}`` cards, and a ``shape="deck"`` region's parts'
    solid volume loaded from its own ``region_rad`` deck (relative to *case_dir*).

    A growth region may name a ``/BOX/...`` card authored in the pre-processor
    (``deck_box_id``) instead of literal coordinates; here that region's ``shape``
    and coordinates (and, for a ``/BOX/RECTA`` with a ``/SKEW/FIX`` skew, its local
    frame) are filled from :meth:`oropt.deck.Deck.box`, so everything downstream —
    selection, guards, the overlay — treats it exactly like a coordinate region.
    A ``shape="deck"`` region is resolved by
    :func:`oropt.keepout.resolve_deck_region` (its part geometry attached for the
    centroid-in-part membership test). Plain coordinate regions are returned
    unchanged. Raises ``ValueError`` when a referenced ``/BOX`` card is absent, or a
    deck region's deck is blank/missing/unparsable."""
    out = []
    for i, b in enumerate(boxes or []):
        label = b.name or f"#{i + 1}"
        if b.shape_kind() == "deck":
            out.append(resolve_deck_region(b, case_dir, label))
            continue
        if getattr(b, "deck_box_id", None) is None:
            out.append(b)
            continue
        spec = deck.box(b.deck_box_id)
        if spec is None:
            raise ValueError(
                f"growth box {label!r} references box id {b.deck_box_id} but no "
                "/BOX/RECTA, /BOX/SPHER or /BOX/CYLIN card with that id is in the "
                "deck; author it in the pre-processor or give literal coordinates")
        out.append(dataclasses.replace(b, deck_box_id=None, **spec))
    return out


#: log/preview note for a carve-off region running without an id boundary —
#: nothing is identifiable as "original part", so the region carves after all.
_NO_BOUNDARY_NOTE = (
    "carve is off but model.growth_original_elem_max is not set -- no "
    "original/expansion element-id boundary is known, so every in-region "
    "element starts void (a part overlap is carved, as if carve were true). "
    "The growth-mesh step records the boundary when pointing the config at "
    "the extended decks; set it manually for a hand-pre-meshed deck")


def _no_boundary(box, model) -> bool:
    """True for a carve-off region with no original/expansion id boundary —
    the (warned) degrade-to-carving case."""
    return (not getattr(box, "carve", False)
            and getattr(model, "growth_original_elem_max", None) is None)


def region_candidate_mask(deck: Deck, mesh: Mesh, box, model
                          ) -> tuple[np.ndarray, int]:
    """``(candidate mask, kept-alive count)`` for a single resolved region.

    The mask is the centroid-in-region element set; a region with ``carve``
    off (the default) additionally excludes the *original* part elements —
    ids <= ``model.growth_original_elem_max`` — so an overlapping region
    leaves the part intact and only its expansion elements start void. The
    kept-alive count is how many in-region elements that exclusion spared (0
    for a carving region). When ``carve`` is off but no id boundary is
    configured, nothing can be told apart from the part, so the region
    degrades to carving — full in-region mask, kept-alive 0 — and the caller
    surfaces :data:`_NO_BOUNDARY_NOTE` (run log / preview / validation).
    The single source of truth for the run-start guard, the preview and the
    PREPARE re-check."""
    bm = mesh.in_boxes_mask([box])
    thr = getattr(model, "growth_original_elem_max", None)
    if getattr(box, "carve", False) or thr is None:
        return bm, 0
    keep_alive = bm & (deck.elem_ids <= int(thr))
    return bm & (deck.elem_ids > int(thr)), int(keep_alive.sum())


def growth_forbid_mask(deck: Deck, mesh: Mesh, model, boxes=None
                       ) -> Optional[np.ndarray]:
    """Forbidden growth space as an element mask (aligned with ``deck.elem_ids``):
    the union of the keep-out deck's neighbour parts
    (``model.growth_keepout_rad``) and every NEGATIVE (``forbid=True``) growth
    box. A candidate whose centroid lands in this space is held void every
    iteration — never grown — so the optimiser adds no material there. Returns
    ``None`` when nothing forbids growth (no keep-out deck and no negative box),
    the fast path.

    *boxes* may be passed already resolved (:func:`resolve_growth_boxes`) to
    avoid resolving twice; omitted, the model's active boxes are resolved here.
    """
    if boxes is None:
        boxes = resolve_growth_boxes(deck, active_growth_boxes(model),
                                     getattr(model, "case_dir", "."))
    keepout = resolve_keepout(model, getattr(model, "case_dir", "."))
    block = keepout.block_mask(mesh.centroids) if keepout is not None else None
    neg = [b for b in (boxes or []) if getattr(b, "forbid", False)]
    if neg:
        nb = mesh.in_boxes_mask(neg)
        block = nb if block is None else (block | nb)
    return block


def growth_candidate_mask(deck: Deck, mesh: Mesh, model,
                          log: Callable[[str], None] = print) -> np.ndarray:
    """Boolean mask (aligned with ``deck.elem_ids``) of growth-candidate
    elements: those whose centroid lies inside a POSITIVE ``model.growth_boxes``
    region (minus, for a region with ``carve`` off, the original part's
    elements — see :func:`region_candidate_mask`). Candidates start the run
    *void* and may be grown into by the optimiser's bi-directional update.
    NEGATIVE (``forbid=True``) regions generate no candidates. All-False when
    no positive boxes are configured.

    A configured **keep-out** deck (``model.growth_keepout_rad``) or any
    **negative box** subtracts the candidates inside the forbidden space from
    the *growable* set: they still start void (returned in the mask, so the
    initial deck omits them) but are held void every iteration by the loop
    (:func:`growth_blocked_mask`), so no material grows there. The run-start
    guards below run only on the growable candidates — a held-void candidate
    needs neither pinnable nodes nor a growth path.

    Raises ``ValueError`` at run start — before any ~13-min solve — for the
    setup mistakes that would otherwise waste (or silently no-op) a multi-hour
    run:

    * a box selecting **no** design elements: the box volume was not pre-meshed
      into the design part (or, with carve off, holds nothing but original
      part elements);
    * candidate connectivity referencing node ids below ``design_node_min``: the
      free-node guard could not pin those nodes while the elements are void, so
      the implicit tangent would go singular;
    * candidates **unreachable** from the initial structure (no shared-node path,
      even through other candidates): a non-conformal interface —
      ``keep_connected`` would drop anything grown there as a floating island.
    """
    boxes = resolve_growth_boxes(deck, active_growth_boxes(model),
                                 getattr(model, "case_dir", "."))
    if not boxes:
        return np.zeros(deck.n_design_elements, dtype=bool)
    # Forbidden growth space = the keep-out deck's neighbour parts PLUS any
    # NEGATIVE (forbid=True) growth box (an inline keep-out). A positive-box
    # candidate landing here starts void like any candidate but is held void
    # every iteration -- no material is ever added inside it.
    block = growth_forbid_mask(deck, mesh, model, boxes)

    candidate = np.zeros(deck.n_design_elements, dtype=bool)  # void-start set
    growable = np.zeros(deck.n_design_elements, dtype=bool)   # minus forbidden
    blocked = np.zeros(deck.n_design_elements, dtype=bool)    # held-void forbidden
    box_growable = []
    for i, b in enumerate(boxes):
        label = b.name or f"#{i + 1}"
        if getattr(b, "forbid", False):
            log(f"[oropt] growth box {label!r}: NEGATIVE (forbidden) region -- "
                "no material may be added inside it")
            continue
        bm, kept_alive = region_candidate_mask(deck, mesh, b, model)
        if not bm.any():
            if kept_alive:
                raise ValueError(
                    f"growth box {label!r} contains only original part "
                    f"elements ({kept_alive}, ids <= "
                    f"{model.growth_original_elem_max}) and has carve off -- "
                    "nothing would start void. Generate the expansion mesh "
                    "with the growth-mesh step, or set carve: true for "
                    "deliberate carve-and-regrow")
            raise ValueError(
                f"growth box {label!r} contains no design elements -- the region "
                "volume must be pre-meshed into the design part "
                f"(/TETRA4/{deck.design_part_id}) before material can grow "
                "there, or generated with the growth-mesh step "
                "(python -m oropt.growthmesh / the GUI's Generate button)")
        log(f"[oropt] growth box {label!r}: {int(bm.sum())} candidate elements "
            "start void")
        if kept_alive:
            log(f"[oropt] growth box {label!r}: {kept_alive} in-region original "
                "part elements stay alive (carve off)")
        elif _no_boundary(b, model):
            log(f"[oropt] growth box {label!r}: {_NO_BOUNDARY_NOTE}")
        gm = bm
        if block is not None:
            blk = bm & block
            nb = int(blk.sum())
            if nb:
                gm = bm & ~block
                blocked |= blk
                log(f"[oropt] growth box {label!r}: {nb} candidate(s) inside a "
                    f"keep-out / forbidden region held void (never grown); "
                    f"{int(gm.sum())} growable")
                if not gm.any():
                    log(f"[oropt] growth box {label!r}: WARNING every candidate "
                        "is inside a keep-out / forbidden region -- this region "
                        "can grow nothing")
        candidate |= bm
        growable |= gm
        box_growable.append((label, gm))

    # Void-element nodes must be pinnable by the free-node guard, which only
    # covers design nodes (ids >= design_node_min). Only growable candidates
    # need pinnable nodes -- a held-void keep-out candidate is never grown.
    cand_nodes = deck.elem_conn[growable]
    bad = cand_nodes < deck.design_node_min
    if bad.any():
        raise ValueError(
            f"growth-box candidate elements reference "
            f"{int(np.unique(cand_nodes[bad]).size)} node id(s) below "
            f"design_node_min={deck.design_node_min}; the free-node guard cannot "
            "pin them while the elements are void (singular implicit tangent). "
            "Renumber the expansion-mesh nodes to ids >= design_node_min")

    # Every growable candidate must be connected (via shared nodes, possibly
    # through other growable candidates) to the initially-alive structure, or it
    # can never be grown: keep_connected would drop it as a floating island every
    # iteration. Held-void keep-out candidates never conduct, so they are
    # excluded from the reachability graph.
    alive0 = ~candidate
    reachable = mesh.keep_connected(~blocked, alive0)
    unreachable = growable & ~reachable
    if unreachable.any():
        names = [label for label, gm in box_growable if (gm & unreachable).any()]
        raise ValueError(
            f"{int(unreachable.sum())} growth-box candidate element(s) in "
            f"box(es) {', '.join(repr(n) for n in names)} share no nodes with "
            "the initial structure (directly or through other candidates), so "
            "they could never be grown. Make the expansion mesh node-conformal "
            "with the part (imprint the part surface, then merge/equivalence "
            "the coincident interface nodes)")
    return candidate


def growth_blocked_mask(deck: Deck, mesh: Mesh, model) -> np.ndarray:
    """Growth candidates that fall inside forbidden growth space — the keep-out
    geometry (``model.growth_keepout_rad``) or any NEGATIVE (``forbid=True``)
    growth box: they start void like any candidate but are **held void every
    iteration** so the optimiser can never place material there. All-False when
    no positive growth boxes, or nothing forbidding growth, is configured. The
    loop re-applies this after each optimiser update (and the auto-mesh PREPARE
    step simply never generates candidate tets here), so pre-meshed and
    auto-meshed workflows both honour the keep-out and the negative boxes."""
    boxes = resolve_growth_boxes(deck, active_growth_boxes(model),
                                 getattr(model, "case_dir", "."))
    empty = np.zeros(deck.n_design_elements, dtype=bool)
    if not boxes:
        return empty
    block = growth_forbid_mask(deck, mesh, model, boxes)
    if block is None:
        return empty
    void_start = empty.copy()
    for b in boxes:
        if getattr(b, "forbid", False):
            continue
        bm, _ = region_candidate_mask(deck, mesh, b, model)
        void_start |= bm
    return void_start & block


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
    notice: str = ""        # config-level caveat (carve off with no id boundary)
    group_guard: str = ""   # /GRNOD group-id guard error (typo'd bc/freeze/stress-exclude id), if any
    keepout: str = ""       # keep-out summary (parts / clearance / held-void count) or its error


def preview_growth_boxes(deck: Deck, mesh: Mesh, model) -> GrowthPreview:
    """Count the design elements each growth region would start *void*, without
    launching a run — the data behind the GUI's "preview regions" button.

    Per POSITIVE region: the would-start-void element count — centroid-in-region,
    minus the original part elements for a region with ``carve`` off (0 flags a
    region whose volume was not pre-meshed into the design part), plus a note for
    a region referencing a ``/BOX/RECTA`` card absent from the deck and for
    in-region original elements a carve-off region leaves alive. A NEGATIVE
    (``forbid=True``) region gets a row noting it is forbidden growth space and
    how many candidates it holds void. When carve-off regions run without an id
    boundary (they degrade to carving), the single config-level ``notice``
    carries the caveat instead of repeating it on every row. Also runs the real
    run-start guards (:func:`growth_candidate_mask`) and reports, in ``guard``, the
    first error that would abort a run (an empty region, candidate nodes below
    ``design_node_min``, or an unreachable candidate); ``guard`` is ``""`` when the
    regions are run-ready, with ``total_candidates`` the unique element count that
    would then start void. Independently runs the /GRNOD group-id guard
    (:func:`validate_group_ids`) -- even with no growth regions configured -- and
    reports its error in ``group_guard``, so a typo'd keep-out / stress-exclusion /
    BC group id surfaces in the preview panel instead of hours into a run.
    Never raises."""
    boxes = getattr(model, "growth_boxes", []) or []
    # Keep-out geometry (neighbour parts) -- resolved once so each region can
    # report how many of its candidates it would hold void. A broken keep-out
    # deck is surfaced (never raises) so the preview stays informative.
    keepout, keepout_note, ko_mask = None, "", None
    if getattr(model, "growth_keepout_rad", None):
        try:
            keepout = resolve_keepout(model, getattr(model, "case_dir", "."))
            ko_mask = keepout.block_mask(mesh.centroids)
        except (ValueError, OSError) as exc:
            keepout_note = f"keep-out error: {exc}"
    # Pre-pass: the positive void-start union and the negative (forbidden)
    # region, so a positive row can report how many of its candidates are held
    # void and a negative row how many candidates it holds void. Errors here are
    # swallowed and surfaced per-row in the main pass below.
    pos_void = np.zeros(deck.n_design_elements, dtype=bool)
    neg_mask = np.zeros(deck.n_design_elements, dtype=bool)
    for b in boxes:
        try:
            rb = resolve_growth_boxes(deck, [b], getattr(model, "case_dir", "."))[0]
        except ValueError:
            continue
        if getattr(rb, "forbid", False):
            neg_mask |= mesh.in_boxes_mask([rb])
        else:
            try:
                bm, _ = region_candidate_mask(deck, mesh, rb, model)
            except ValueError:
                continue
            pos_void |= bm
    forbid_mask = neg_mask if ko_mask is None else (neg_mask | ko_mask)
    rows: list = []
    for i, b in enumerate(boxes):
        label = b.name or f"#{i + 1}"
        try:
            resolved = resolve_growth_boxes(
                deck, [b], getattr(model, "case_dir", "."))[0]
        except ValueError as exc:
            rows.append(BoxPreview(label, b.shape_kind(), 0, str(exc)))
            continue
        if getattr(resolved, "forbid", False):
            inside = mesh.in_boxes_mask([resolved])
            held = int((inside & pos_void).sum())
            note = ("negative (forbidden) region -- no material may be added "
                    "inside it")
            note += (f"; holds {held} candidate(s) void" if held
                     else " (no growth region overlaps it -- no-op)")
            rows.append(BoxPreview(label, resolved.shape_kind(),
                                   int(inside.sum()), note))
            continue
        try:
            bm, kept_alive = region_candidate_mask(deck, mesh, resolved, model)
        except ValueError as exc:
            rows.append(BoxPreview(label, resolved.shape_kind(), 0, str(exc)))
            continue
        count = int(bm.sum())
        if count:
            note = (f"{kept_alive} in-region original element(s) stay alive "
                    "(carve off)" if kept_alive else "")
        elif kept_alive:
            note = (f"only original part elements inside ({kept_alive}) and "
                    "carve is off -- nothing would start void")
        else:
            note = ("no design elements inside -- the region volume "
                    "is not pre-meshed into the design part")
        if count:
            blk = int((bm & forbid_mask).sum())
            if blk:
                held = ("all of them" if blk == count else f"{blk} of {count}")
                note = (f"{note}; " if note else "") + (
                    f"{held} held void by keep-out / forbidden region "
                    "(never grown)")
        rows.append(BoxPreview(label, resolved.shape_kind(), count, note))
    total, guard = 0, ""
    if boxes:
        try:
            total = int(growth_candidate_mask(
                deck, mesh, model, log=lambda _m: None).sum())
        except ValueError as exc:
            guard = str(exc)
    notice = (_NO_BOUNDARY_NOTE
              if any(_no_boundary(b, model) for b in boxes
                     if not getattr(b, "forbid", False)) else "")
    group_guard = ""
    try:
        validate_group_ids(deck, model)
    except ValueError as exc:
        group_guard = str(exc)
    if keepout is not None and not keepout_note:
        held = int((pos_void & ko_mask).sum()) if ko_mask is not None else 0
        parts = ", ".join(str(p) for p in keepout.part_ids) or "all"
        clr = (f", clearance {keepout.clearance:g}" if keepout.clearance > 0
               else f", allowed penetration {-keepout.clearance:g}"
               if keepout.clearance < 0 else "")
        keepout_note = (
            f"keep-out {Path(keepout.source).name} (part(s) {parts}{clr}): "
            f"{held} candidate(s) held void")
        if not held:
            keepout_note += " -- no growth region overlaps the neighbour parts (no-op)"
    return GrowthPreview(rows=rows, total_candidates=total,
                         total_elements=deck.n_design_elements, guard=guard,
                         notice=notice, group_guard=group_guard,
                         keepout=keepout_note)


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


def case_violation(case, res) -> float | None:
    """One load case's worst constraint-utilisation ratio ``value/limit`` over
    its stress and every displacement constraint — the per-case analogue of
    :func:`worst_violation`. ``None`` when the case configures no limit (nothing
    to equalise on), so the adaptive-weight controller holds its weight and
    excludes it from the mean."""
    ratios = []
    if case.sigma_allow is not None:
        ratios.append(res.sigma_max / case.sigma_allow)
    for _nid, val, limit, _feas in disp_breakdown(case, res):
        if limit is not None:
            ratios.append(val / limit)
    return max(ratios) if ratios else None


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


def _solve_activity(it: int, case: ResolvedCase, i: int, n_cases: int,
                    slot=None) -> str:
    """One-line "what is solving right now" for the live ``Status.activity``.

    Names the iteration, the load case and — the point of the fast-mode monitor —
    whether this solve is the fast tied-linear screen or the full nonlinear solve,
    so the GUI shows what is running during the minutes a solve takes. When *slot*
    (a :class:`~oropt.config.SolverSlot`) is given the solver's own np/nt is
    appended, so a per-slot concurrent run shows each solve's CPU allocation."""
    mode = "fast linear (tied)" if case.fast_mode else "full nonlinear"
    where = f" [case {i + 1}/{n_cases}]" if n_cases > 1 else ""
    cpu = f" (np={int(slot.np)} nt={int(slot.nt)})" if slot is not None else ""
    return f"iter {it}: solving {case.name!r} — {mode}{where}{cpu}"


def iter0_archive_dir(work: Path, stem: str, n_cases: int) -> Path:
    """Where iteration 0's archived solve for *stem* lives (or would be copied to).

    Mirrors :func:`_archive_iteration`'s layout: ``work/iter_0000/`` for a
    single-case run, ``work/iter_0000/<stem>/`` when several cases share a folder."""
    d = work / "iter_0000"
    return d / stem if n_cases > 1 else d


def copy_iter0(src_run: str | Path, dst_run: str | Path,
               overwrite: bool = False) -> tuple[bool, str]:
    """Copy an ``iter_0000`` from *src_run* into *dst_run* to seed a solve reuse.

    Copies the whole ``iter_0000`` tree (so single- and multi-case layouts both
    work), which the loop then validates and reuses at iteration 0 (see
    :func:`reuse_iter0_solve` — the byte-compare of the starter deck means a copy
    from a different model is refused at solve time, so this copy is safe to offer).
    Returns ``(ok, message)``: refuses when the source has no ``iter_0000``, when
    source and destination are the same folder, or when the destination already has
    one and *overwrite* is False."""
    src = Path(src_run) / "iter_0000"
    dst = Path(dst_run) / "iter_0000"
    if not src.is_dir():
        return False, f"no iter_0000 folder in {src_run}"
    if src.resolve() == dst.resolve():
        return False, "source and destination run folders are the same"
    if dst.exists():
        if not overwrite:
            return False, "this run already has an iter_0000 (enable overwrite to replace)"
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True, f"copied iter_0000 from {src_run}"


def reuse_iter0_solve(reuse_dir: Path, solve_dir: Path, stem: str,
                      starter: Path, log: Callable[[str], None] = print
                      ) -> Optional[RunResult]:
    """Reuse an already-present iteration-0 solve instead of re-running it.

    Iteration 0 solves the initial full-volume design — the single most expensive
    solve — and is identical across runs that share the same initial deck. If the
    user drops a matching ``iter_0000`` into the run folder (copied from an earlier
    run), this reuses its animation rather than re-solving.

    Guarded so a wrong copy can never poison the run: the archived starter deck is
    byte-compared against the one this run just wrote (``deck.write`` is
    deterministic, so identical starters ⇒ identical solves). On a match the
    archived ``<stem>A0*`` animation (+ listing) is copied into *solve_dir* and an
    ok :class:`RunResult` is returned; on any mismatch / missing file it returns
    ``None`` and the caller solves fresh. Reason is always logged."""
    archived = reuse_dir / f"{stem}_0000.rad"
    anims = sorted(reuse_dir.glob(f"{stem}A[0-9][0-9]*"))
    if not archived.is_file() or not anims:
        log(f"[oropt] iter 0: {reuse_dir.name}/ has no reusable {stem} solve "
            "(starter/animation missing) -- solving fresh")
        return None
    if not filecmp.cmp(starter, archived, shallow=False):
        log(f"[oropt] iter 0: {reuse_dir.name}/ starter differs from this run's "
            f"initial {stem} design -- solving fresh (copied from another model?)")
        return None
    for src in [*anims, reuse_dir / f"{stem}_0001.out"]:
        if src.is_file():
            shutil.copy2(src, solve_dir / src.name)
    log(f"[oropt] iter 0: REUSING {reuse_dir.name}/ solve for case {stem!r} "
        "-- skipping the initial full-volume solve")
    return RunResult(ok=True, stage="ok", message=f"reused {reuse_dir.name}")


def _solve_case(cfg: Config, case: ResolvedCase, deck: Deck, alive: np.ndarray,
                no_pin: set, solve_dir: Path, anim_dt: float,
                exclude_elem_ids: Optional[np.ndarray] = None,
                fast_tie: Optional[FastModeTie] = None,
                reuse_dir: Optional[Path] = None,
                log: Callable[[str], None] = print):
    """Write the alive deck for one load case, solve it, extract its results.

    The per-case "solve + extract" unit reused for every case each iteration. The
    case's ``stem`` / ``disp_node_id`` are passed straight to ``run_solver`` /
    ``extract`` (they accept them explicitly), so one *cfg* drives every case.
    *exclude_elem_ids* (the stress-exclusion set, shared by all cases) is forwarded
    to ``extract`` so the case's ``sigma_max`` ignores those elements.

    When ``case.fast_mode`` is set the alive starter is turned into a tied-linear
    deck (:func:`oropt.fastmode.build_fast_case`, using the precomputed *fast_tie*)
    and solved with a plain ``/IMPL/LINEAR`` engine instead of the nonlinear one;
    ``extract`` then reads the anim von-Mises exactly as usual. With ``fast_mode``
    off the path is byte-identical to before. Returns ``(run_result, results)``
    where *results* is ``None`` if the solve failed (the caller surfaces
    *run_result*).

    *reuse_dir* (iteration 0 only) is an archived solve to reuse if its starter
    matches this run's — set by :func:`reuse_iter0_solve`. The starter/engine are
    still written first, so the reuse check compares against the exact deck that
    would otherwise be solved (and fast-mode transforms are already applied)."""
    _clean_solve_dir(solve_dir)
    starter = solve_dir / f"{case.stem}_0000.rad"
    engine = solve_dir / f"{case.stem}_0001.rad"
    deck.write(starter, alive, no_pin=no_pin)
    if getattr(cfg, "demo", None) is not None and cfg.demo.enabled:
        # Demo backend: answer the solve with deterministic synthetic physics
        # (oropt.demo) — no starter/engine/anim, no OpenRadioss install. The
        # deck is still written above so the deletion/pinning path stays
        # exercised; everything downstream (feasibility, update, status,
        # post-processing) sees a normal (RunResult, Results) pair.
        from .demo import demo_solve
        return demo_solve(deck, alive, case, cfg.demo)
    if case.fast_mode:
        if fast_tie is None:                     # precomputed in run_optimization
            raise ValueError(f"fast-mode case {case.name!r} has no discovered tie")
        build_fast_case(deck, alive, starter, case.engine, engine, fast_tie,
                        anim_dt=anim_dt)
    else:
        # sensitivity: tdsa needs the per-element stress tensor in the anim —
        # inject /ANIM/BRICK/TENS/STRESS so extraction can feed the topological-
        # derivative ranking (absent tensor -> map_sensitivity falls back to
        # energy with a warning). Other modes keep the engine deck byte-identical.
        want_tensor = getattr(cfg.active_opts(), "sensitivity", "energy") == "tdsa"
        prepare_engine(case.engine, engine, anim_dt=anim_dt,
                       anim_stress_tensor=want_tensor)
    res = (reuse_iter0_solve(reuse_dir, solve_dir, case.stem, starter, log)
           if reuse_dir is not None else None)
    if res is None:
        res = run_solver(cfg, solve_dir, stem=case.stem)
    if not res.ok:
        return res, None
    return res, extract(cfg, solve_dir, stem=case.stem,
                        disp_node_ids=[dc.node_id for dc in case.disp_constraints],
                        exclude_element_ids=exclude_elem_ids)


def _sequential_contract(results: list) -> tuple[list, list]:
    """``(run_results, case_results)`` from per-case ``(run_result, results)`` in
    case order, stopping at the first failed case.

    Reproduces the sequential loop's shape regardless of how the solves were
    dispatched: ``case_results`` is the successful prefix, ``run_results`` ends at
    the first failure (``results is None``). The downstream failure handling then
    reads ``run_results[-1]`` as the failure and ``cases[len(case_results)]`` as the
    case that failed — unchanged whether cases ran one at a time or concurrently."""
    run_results: list = []
    case_results: list = []
    for res, r in results:
        run_results.append(res)
        if r is None:
            break
        case_results.append(r)
    return run_results, case_results


def _slot_cfg(cfg: Config, slot) -> Config:
    """A shallow :class:`~oropt.config.Config` copy whose ``run`` and ``docker``
    np/nt are *slot*'s, so :func:`oropt.runner.run_solver` uses this concurrent
    solver's own CPU allocation (native reads ``run.np``/``run.nt``, Docker reads
    ``docker.np``/``docker.nt`` — both overridden). Every other setting is shared
    by reference (read-only during the solve)."""
    npv, ntv = int(slot.np), int(slot.nt)
    return dataclasses.replace(
        cfg, run=dataclasses.replace(cfg.run, np=npv, nt=ntv),
        docker=dataclasses.replace(cfg.docker, np=npv, nt=ntv))


def _slot_plan(cfg: Config, n_cases: int) -> tuple[int, list]:
    """``(concurrency, per_case_slot)`` for this iteration's solves.

    With ``run.solver_slots`` configured, the concurrency is the number of slots
    (clamped to *n_cases*) and case *i* is assigned slot ``i % concurrency``;
    ``per_case_slot`` is the list of those slots (``SolverSlot`` or ``None``) per
    case, so a caller can read each solve's CPU allocation. With no slots the
    concurrency is ``run.solver_concurrency`` and every entry is ``None`` (the
    global ``run.np``/``run.nt`` apply, unchanged behaviour)."""
    slots = list(getattr(cfg.run, "solver_slots", []) or [])
    if slots:
        conc = max(1, min(len(slots), n_cases))
        return conc, [slots[i % conc] for i in range(n_cases)]
    conc = max(1, min(int(getattr(cfg.run, "solver_concurrency", 1)), n_cases))
    return conc, [None] * n_cases


def _solve_cases(cfg: Config, cases, case_decks, alive, no_pin, solve_root,
                 n_cases, exclude_elem_ids, fast_ties, reuse_dirs, status, work,
                 it, log) -> tuple[list, list]:
    """Solve every load case for iteration *it* and return the sequential-contract
    ``(run_results, case_results)``.

    Concurrency comes from ``run.solver_slots`` (each slot its own np/nt) if set,
    else ``run.solver_concurrency`` (uniform ``run.np``/``run.nt``), clamped to
    ``[1, n_cases]``. Each case solves in its own ``solve/case_<i>/`` dir (see
    :func:`_case_solve_dir`) as an independent subprocess; ``concurrency == 1``
    keeps the exact sequential path (per-case live activity, stop at the first
    failure). >1 without slots uses a thread pool over the cases (dynamic load
    balancing). >1 **with** slots runs one worker per slot, each solving its
    round-robin subset of cases SERIALLY, so a slot's np/nt is never used by two
    solves at once — the machine load stays capped at the sum over slots. Every
    path reconstructs the same contract, so downstream handling is identical."""
    conc, per_slot = _slot_plan(cfg, n_cases)
    slotted = any(s is not None for s in per_slot)

    def solve_one(i: int):
        slot = per_slot[i]
        c = cfg if slot is None else _slot_cfg(cfg, slot)
        return _solve_case(c, cases[i], case_decks[i], alive, no_pin,
                           _case_solve_dir(solve_root, n_cases, i), c.run.anim_dt,
                           exclude_elem_ids, fast_tie=fast_ties[i],
                           reuse_dir=reuse_dirs[i], log=log)

    if conc == 1:
        results: list = []
        for i in range(n_cases):
            status.state = "running"; status.iteration = it
            status.activity = _solve_activity(it, cases[i], i, n_cases, per_slot[i])
            st.write_status(work, status)
            res, r = solve_one(i)
            results.append((res, r))
            if r is None:                        # stop at the first failure
                break
        return _sequential_contract(results)

    status.state = "running"; status.iteration = it
    if slotted:
        alloc = ", ".join(f"slot{k}: np={int(per_slot[k].np)} nt={int(per_slot[k].nt)}"
                          for k in range(conc))
        status.activity = (f"iter {it}: solving {n_cases} load cases across "
                           f"{conc} solver slots ({alloc})")
    else:
        status.activity = (f"iter {it}: solving {n_cases} load cases, "
                           f"{conc} concurrently")
    st.write_status(work, status)
    with ThreadPoolExecutor(max_workers=conc) as ex:
        if slotted:
            # One worker per slot, each solving its cases serially -> a slot's
            # np/nt is never doubled up (total load = sum over active slots).
            def slot_worker(slot_idx: int) -> dict:
                out: dict = {}
                for i in range(slot_idx, n_cases, conc):
                    out[i] = solve_one(i)
                return out
            merged: dict = {}
            for fut in [ex.submit(slot_worker, k) for k in range(conc)]:
                merged.update(fut.result())        # re-raises worker exceptions
            results = [merged[i] for i in range(n_cases)]
        else:
            futs = {i: ex.submit(solve_one, i) for i in range(n_cases)}
            results = [futs[i].result() for i in range(n_cases)]  # order; re-raises
    return _sequential_contract(results)


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


def _archived_iter_dir(work: Path, it: int, stem: str, n_cases: int) -> Path:
    """The folder :func:`_archive_iteration` wrote iteration *it*'s files into
    (mirrors its layout): ``work/iter_NNNN/`` for a single case,
    ``work/iter_NNNN/<stem>/`` when there are multiple load cases."""
    d = work / f"iter_{it:04d}"
    return d / stem if n_cases > 1 else d


def _final_anim_dir(work: Path, feas_it: int, stem: str, n_cases: int,
                    fallback: Path) -> tuple[Path, bool]:
    """Which directory's ``<stem>A0*`` animation to convert to d3plot.

    Returns ``(dir, used_archive)`` — the last feasible iteration's *archived*
    animation (``work/iter_NNNN/`` via :func:`_archive_iteration`) when it exists,
    else *fallback* (the last solved iteration's solve dir). Near convergence the
    optimiser oscillates across the constraint boundary, so the last *solved*
    iteration held by *fallback* is often infeasible; using the archived feasible
    iteration makes the d3plot show the same last-feasible design the report renders.
    The fallback keeps the old behaviour when per-iteration archiving is off or the
    feasible iteration's animation wasn't archived (e.g. an aborted solve)."""
    if feas_it >= 0:
        arch = _archived_iter_dir(work, feas_it, stem, n_cases)
        if sorted(arch.glob(f"{stem}A0*")):
            return arch, True
    return fallback, False


def _converged(vfs: list[float], feasible: bool, target_vf: float,
               window: int, tol: float) -> bool:
    if not feasible or len(vfs) < window:
        return False
    if vfs[-1] > target_vf * (1.0 + 1e-6):
        return False
    recent = vfs[-window:]
    return (max(recent) - min(recent)) / max(np.mean(recent), 1e-12) < tol


def _reset_stale_outputs(work: Path, log: Callable[[str], None]) -> None:
    """Clear a previous run's *accumulating* outputs before a fresh (non-resume)
    run into the same folder, so the new run's history and evolution are not mixed
    with the old one's.

    ``history.csv`` is appended per iteration and the ``topology_iter*`` /
    ``topology_smoothed_iter*`` snapshots accumulate, so re-running fresh into a
    used folder would otherwise prepend the previous run's rows/frames (a killed
    dead run's iterations showing up before the real ones). This mirrors
    :func:`oropt.run._tee_log` truncating ``run.log`` on a fresh start.

    Preserved: the ``iter_NNNN/`` archives (a seeded ``iter_0000`` is byte-checked
    before reuse, not blindly trusted — see :func:`reuse_iter0_solve`), and
    ``checkpoint.npz`` / ``status.json`` (overwritten in-place as the run proceeds).
    Single-file end products (``report.*``, ``topology_evolution.gif``,
    ``topology_latest.vtu``) are rewritten wholesale, so they never mix.
    """
    removed = 0
    hist = work / st.HISTORY
    if hist.exists():
        hist.unlink(missing_ok=True)
        removed += 1
    for pat in ("topology_iter*.vtu", "topology_smoothed_iter*.stl"):
        for p in work.glob(pat):
            p.unlink(missing_ok=True)
            removed += 1
    if removed:
        log(f"[oropt] fresh run: cleared {removed} stale output file(s) from a "
            "previous run in this folder (history + per-iteration snapshots); "
            "iteration archives and checkpoint are kept")


def run_optimization(cfg: Config, resume: bool = False,
                     log: Callable[[str], None] = print,
                     should_stop: Optional[Callable[[], bool]] = None) -> st.Status:
    work = cfg.work()
    solve_root = work / "solve"
    m = cfg.model

    # A resume without a checkpoint (e.g. the prior run crashed during iteration
    # 0, before the first save) would silently start from scratch *while keeping*
    # the old history/snapshots -- duplicated iteration numbers, and the post-run
    # last-feasible selection could pick the OLD run's design. Downgrade to a
    # fresh run loudly, so the stale-output reset below applies.
    if resume and not (work / st.CHECKPOINT).exists():
        log(f"[oropt] resume requested but no {st.CHECKPOINT} in {work} -- "
            "starting FRESH instead (previous history/snapshots are cleared)")
        resume = False

    # Snapshot the exact config this run uses into its own run folder so every
    # result set carries the config that produced it; a resume also preserves the
    # prior stage's snapshot and reports the optimiser it used (for a switch flag).
    prior_optimizer = snapshot_config_used(work, cfg, resume, log)

    (work / "stop.flag").unlink(missing_ok=True)   # ignore any stale stop request
    if not resume:                                   # fresh run: don't let a prior
        _reset_stale_outputs(work, log)              # run's history/snapshots mix in
    # Claim the run dir NOW, not after setup: deck parsing + mesh/filter building
    # take minutes on production meshes, and until run.pid exists `is_running` is
    # False -- the whole window the GUI's Start button and the queue runner's
    # busy-wait use to decide a second launch into this folder is safe.
    pid = st.write_pid(work)
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
    validate_group_ids(deck, m)   # typo'd /GRNOD ids fail here, not hours in
    case_decks = [deck]
    for case in cases[1:]:
        cdeck = Deck.load(case.starter, m.design_part_id, m.design_node_min)
        if not np.array_equal(cdeck.elem_ids, deck.elem_ids):
            raise ValueError(
                f"load case {case.name!r} (stem {case.stem!r}) has a different "
                f"design-part element set than the primary case {primary.name!r}; "
                "all load cases must share the same mesh")
        case_decks.append(cdeck)
    # Fast-mode ties: discovered once per fast-mode case (parsing the full deck is
    # a few seconds, so it is not repeated every iteration). The load/support tie
    # patches sit in protected regions -> they stay alive, so the tie is stable
    # across iterations; _solve_case re-intersects it with the alive mesh each time
    # via build_fast_case. Non-fast cases keep a None entry and solve as before.
    fast_ties: list[Optional[FastModeTie]] = [None] * n_cases
    for i, (case, cdeck) in enumerate(zip(cases, case_decks)):
        if case.fast_mode:
            tie = discover_tie(cdeck, m.design_node_min)
            fast_ties[i] = tie
            lim = "unset" if case.sigma_allow is None else f"{case.sigma_allow:g} MPa"
            log(f"[oropt] case {case.name!r}: FAST MODE (tied linear screen, "
                f"sigma_allow={lim}); {tie.summary()}")
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
    # Forbidden growth space (keep-out neighbour parts and/or negative "forbid"
    # boxes): candidates inside it start void like any candidate but are held
    # void every iteration (never grown), so the optimiser can never place
    # material there. Re-applied after every optimiser update via ``keep_growable``.
    blocked = growth_blocked_mask(deck, mesh, m)
    keep_growable = ~blocked
    hold_void = bool(blocked.any())
    if n_candidate:
        overlap = int((protected & candidate).sum())
        if overlap:
            log(f"[oropt] growth: {overlap} candidate elements overlapped the "
                "protected set -> left unprotected (candidates are never frozen)")
        protected = protected & ~candidate
        log(f"[oropt] growth: {n_candidate} candidate elements "
            f"({100 * candidate.mean():.1f}% of the design space) start void; "
            "volume fractions are relative to the enlarged (part + boxes) space")
        if hold_void:
            log(f"[oropt] growth: {int(blocked.sum())} candidate(s) inside a "
                "keep-out / forbidden region held void every iteration (never "
                "grown) -> no material is added there")
    opt = build_optimizer(cfg, mesh, protected, anchor=anchor)
    # Multipoint back-off (backoff_mode: multipoint): the volume target comes
    # from a controller fitted to the run's own (vf, violation) history instead
    # of the optimiser's reactive gate. None = the classic gate, unchanged.
    ctrl = build_backoff_controller(oc)
    # Adaptive per-load-case weights (run.adaptive_weights): the combined
    # sensitivity's per-case split is nudged toward equal constraint utilisation
    # each iteration. None = fixed configured weights, byte-identical to before.
    wctrl = build_weight_controller(cfg, [c.weight for c in cases])
    log(f"[oropt] optimizer={cfg.optimizer_name()}; protected elements: "
        f"{int(protected.sum())} ({100*protected.mean():.1f}%); V0={opt.V0:.3f}"
        + ("; backoff=multipoint" if ctrl is not None else "")
        + ("; adaptive-weights" if wctrl is not None else ""))

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
            # Restore the level-set's nodal field: re-initialising phi from the
            # mask would silently perturb the design and re-order the field by
            # the current sensitivity rank. Other optimisers carry no phi.
            phi = ckpt["phi"]
            if phi is not None and hasattr(opt, "phi"):
                if phi.shape == (mesh.n_nodes,):
                    opt.phi = phi
                else:
                    log(f"[oropt] checkpoint phi has {phi.size} nodes but the "
                        f"mesh has {mesh.n_nodes} -- ignored, phi will "
                        "re-initialise from the alive mask")
            # HCA's per-element virtual density. Restored only into an optimiser
            # that carries one (hasattr .x) and only when the element count matches
            # -- a nodal phi never fits here, so a level-set -> HCA switch correctly
            # falls through to a re-init from the alive mask.
            xfield = ckpt.get("x")
            if xfield is not None and hasattr(opt, "x"):
                if xfield.shape == (deck.n_design_elements,):
                    opt.x = xfield
                else:
                    log(f"[oropt] checkpoint density field has {xfield.size} "
                        f"elements but the deck has {deck.n_design_elements} -- "
                        "ignored, density will re-initialise from the alive mask")
            # Multipoint controller history: restored so the resumed run keeps
            # its fitted boundary instead of re-learning it from gate steps.
            if ctrl is not None:
                ctrl.restore(ckpt.get("ctrl"))
            # Adaptive per-load-case weights: restored so a resume continues the
            # adapted split instead of resetting to the configured weights.
            if wctrl is not None:
                wctrl.restore(ckpt.get("weights"))
            cur_vf = opt.volume_fraction(alive)
            log(f"[oropt] resumed at iteration {start_iter}, vf={cur_vf:.3f}")
            for msg in resume_warnings(prior_optimizer, cfg.optimizer_name(),
                                       cur_vf, oc):
                log(msg)
    if hold_void:
        # A resumed (or pre-keep-out) checkpoint may carry material inside the
        # neighbour parts; hold it void from the first iteration on.
        alive = alive & keep_growable

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
    # Whole-run wall-clock budget (run.max_wall_hours, 0 = off). Measured from
    # here — setup included — so it tracks what a cluster/session limit would
    # see. Checked only at iteration boundaries: a solve in flight is never
    # killed, and the checkpoint written at the end of each iteration makes the
    # stop cleanly resumable.
    run_t0 = time.time()
    budget_s = max(0.0, float(cfg.run.max_wall_hours)) * 3600.0
    consecutive_diverged = 0     # non-converged iterations in a row (watchdog)
    # Controller-stall guards (warn-only): a grow request that still loses
    # volume means something outside the volume controller is eating material
    # (the level-set prune-leak signature), and a removal spike far above the
    # recent per-iteration rate is the plateau-collapse signature -- both were
    # measured on the 2026-07-05 elevator-linkage run (docs/levelset_stuck_analysis.md).
    prev_vf: Optional[float] = None          # vf the last target was computed from
    prev_target_vf: Optional[float] = None   # what the gate asked of the last update
    grow_stall = 0                           # consecutive grow-yet-shrink iterations
    removal_hist: list[int] = []             # recent per-update removal counts
    try:
        for it in range(start_iter, oc.max_iter):
            if should_stop and should_stop():
                status.state = "stopped"; status.message = "stop requested"
                break
            # Tell schedule-carrying optimisers (HCA's MHCA neighbourhood decay)
            # the absolute iteration, so a --resume continues the schedule where
            # it left off instead of restarting it wide.
            if hasattr(opt, "set_iteration"):
                opt.set_iteration(it)
            if budget_s > 0.0 and (time.time() - run_t0) >= budget_s:
                wall_h = (time.time() - run_t0) / 3600.0
                status.state = "stopped"
                status.message = (
                    f"wall-clock budget reached ({wall_h:.2f} h >= "
                    f"run.max_wall_hours={cfg.run.max_wall_hours:g} h)")
                log(f"[oropt] {status.message} -- stopping cleanly "
                    "(continue later with --resume)")
                break

            log(f"[oropt] iter {it}: vf={opt.volume_fraction(alive):.3f} "
                f"alive={int(alive.sum())} -> solving "
                + (f"{n_cases} load cases ..." if n_cases > 1 else "..."))
            # ---- solve every load case (up to run.solver_concurrency at once,
            # each in its own dir). Iteration 0 only: reuse an already-present
            # (e.g. copied) matching iter_0000 solve per case instead of re-running
            # the full-volume solve.
            t0 = time.time()
            reuse_dirs = [iter0_archive_dir(work, case.stem, n_cases)
                          if it == 0 and cfg.run.reuse_iter0 else None
                          for case in cases]
            run_results, case_results = _solve_cases(
                cfg, cases, case_decks, alive, no_pin, solve_root, n_cases,
                exclude_elem_ids, fast_ties, reuse_dirs, status, work, it, log)
            iter_wall = time.time() - t0
            elapsed += iter_wall

            res = run_results[-1]
            if not res.ok and res.diverged:
                # ---- non-converged solve: INFEASIBLE, back off, carry on ---
                # The engine watchdog killed a diverging implicit solve (severed
                # load path, collapsing timestep). Its partial outputs are never
                # parsed into the sensitivity; the iteration is treated like a
                # violated constraint instead: the gate raises the volume target
                # (worst violation -> the proportional back-off hits its cap)
                # and the previous iteration's sensitivity re-grows material
                # from the still-intact alive mask.
                failed = cases[len(case_results)]
                consecutive_diverged += 1
                vf = opt.volume_fraction(alive)
                infeasible_msg = (f"case {failed.name!r}: " if n_cases > 1 else "") \
                    + "engine did not converge -- treated as infeasible"
                log(f"[oropt] iter {it}: case {failed.name!r} did not converge "
                    f"({res.message}); treated as INFEASIBLE, backing off "
                    f"({consecutive_diverged}/{cfg.run.diverge_fail_after} "
                    "consecutive)")
                if consecutive_diverged >= cfg.run.diverge_fail_after:
                    status.state = "failed"
                    status.message = (
                        f"{consecutive_diverged} consecutive iterations did not "
                        "converge (run.diverge_fail_after="
                        f"{cfg.run.diverge_fail_after}) -- last: {infeasible_msg}")
                    status.iteration = it
                    status.or_termination = res.message
                    log(f"[oropt] SOLVE FAILED: {status.message}")
                    break
                vfs.append(vf)
                status = st.Status(
                    state="running", iteration=it, max_iter=oc.max_iter,
                    volume_fraction=vf,
                    sigma_allow=(primary.sigma_allow if primary.sigma_allow
                                 is not None else float("nan")),
                    d_allow=(min(primary_d_limits) if primary_d_limits
                             else float("nan")),
                    feasible=False, elements_alive=int(alive.sum()),
                    elements_total=deck.n_design_elements,
                    stress_excluded_elems=n_excluded,
                    elements_candidate=n_candidate,
                    elements_grown=int((alive & candidate).sum()),
                    or_termination=res.message, iter_wall_s=iter_wall,
                    elapsed_s=elapsed, eta_s=iter_wall * (oc.max_iter - it - 1),
                    message=infeasible_msg, pid=pid)
                st.write_status(work, status)
                st.append_history(work, {
                    "iteration": it, "volume_fraction": round(vf, 6),
                    "sigma_max": float("nan"), "disp": float("nan"),
                    "elements_alive": int(alive.sum()), "feasible": False,
                    "iter_wall_s": round(iter_wall, 1),
                    "or_termination": res.message,
                    "optimizer": cfg.optimizer_name()})
                # Without a previous sensitivity (very first iteration) there is
                # nothing to re-grow with; the mask stays put and a
                # deterministic re-diverge fails the run via diverge_fail_after.
                if sens_prev is not None:
                    # A diverged solve carries no usable (vf, violation) point;
                    # the multipoint controller sees the inf and falls back to
                    # the gate's capped back-off, unrecorded.
                    target_vf = (ctrl or opt).next_target_vf(
                        vf, False, violation=float("inf"))
                    alive = opt.update(alive, sens_prev, target_vf)
                    if hold_void:
                        alive = alive & keep_growable
                st.save_checkpoint(work, it + 1, alive, sens_prev,
                                   phi=getattr(opt, "phi", None),
                                   x=getattr(opt, "x", None),
                                   ctrl=(ctrl.state() if ctrl else None),
                                   weights=(wctrl.state() if wctrl else None))
                # the recovery step invalidates the guards' iteration pairing
                prev_vf = prev_target_vf = None
                continue
            if not res.ok:
                failed = cases[len(case_results)]
                status.state = "failed"
                status.message = (f"{res.stage}: {res.message}" if n_cases == 1
                                  else f"case {failed.name!r}: {res.stage}: {res.message}")
                status.iteration = it
                status.or_termination = res.message
                log(f"[oropt] SOLVE FAILED: {status.message}")
                break
            consecutive_diverged = 0

            # ---- null-solve guard -----------------------------------------
            # A solve can reach NORMAL TERMINATION yet carry no load: zero stress
            # AND zero strain energy across the whole design part (the model never
            # deformed). The load landed on a constrained / rigid DOF, a contact
            # never engaged, or the deck was mis-exported. Every feasibility metric
            # then reads 0 and passes trivially and the energy sensitivity is
            # uniformly zero, so the optimiser would silently strip the part to its
            # protected skeleton -- exactly how opti_run5_Ti lost continuity. Fail
            # loudly instead of "optimising" a dead model. Checked every iteration;
            # at iter 0 (full solid) a real load MUST develop stress, so it is
            # unambiguous, but a mid-run all-zero solve is just as meaningless.
            null_cases = [cases[i].name for i, r in enumerate(case_results)
                          if r.is_null_solve]
            if null_cases:
                which = ", ".join(repr(n) for n in null_cases)
                status.state = "failed"
                status.iteration = it
                status.or_termination = res.message
                status.message = (
                    f"null solve at iter {it}: case(s) {which} produced zero "
                    "stress and zero strain energy over the design part -- the "
                    "model carried no load. Check the load path (force applied to "
                    "a constrained or rigid DOF, a contact that never engaged, or "
                    "a mis-exported deck); refusing to optimise a model under no "
                    "load.")
                st.write_status(work, status)
                log(f"[oropt] SOLVE FAILED (null solve): {status.message}")
                break

            # ---- combine cases: weighted-sum sensitivity + worst-case gate -
            raws = [opt.raw_sensitivity(r, deck.elem_ids, alive)
                    for r in case_results]
            weights = (wctrl.weights if wctrl is not None
                       else [c.weight for c in cases])
            raw = combine_sensitivity(raws, weights)
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
                    "fast_mode": bool(case.fast_mode),
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
            # Adaptive weights: step the per-case split from this iteration's
            # per-case utilisation ratios so the new weights apply next combine.
            if wctrl is not None:
                new_w = wctrl.update([case_violation(case, r)
                                      for case, r in zip(cases, case_results)])
                log("[oropt] adaptive weights -> "
                    + ", ".join(f"{c.name}={w:.3f}" for c, w in zip(cases, new_w)))
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
            grow_stall = _grow_stall(grow_stall, prev_target_vf, prev_vf, vf)
            if grow_stall >= GROW_STALL_ITERS:
                log(f"[oropt] WARNING: volume fell {grow_stall} iterations in "
                    "a row against a GROW target -- the volume controller is "
                    "being outrun by removal outside its accounting (see "
                    "docs/levelset_stuck_analysis.md); the run is likely "
                    "ratcheting away from feasibility, not converging")

            # ---- publish state for the GUI --------------------------------
            remaining = oc.max_iter - it - 1
            status = st.Status(
                state="running", iteration=it, max_iter=oc.max_iter,
                volume_fraction=vf, sigma_max=sigma_max,
                sigma_allow=sigma_allow, disp=disp,
                d_allow=d_allow, feasible=feasible,
                elements_alive=int(alive.sum()),
                elements_total=deck.n_design_elements,
                design_volume=float(opt.V0),
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
                "iter_wall_s": round(iter_wall, 1), "or_termination": res.message,
                "optimizer": cfg.optimizer_name()})
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
                # Keep the checkpoint's `it + 1` invariant on this exit too:
                # iteration `it` fully completed (solved, published, archived),
                # so an extend-max_iter resume must start AFTER it -- not
                # re-solve the identical converged design and append a
                # duplicate history row.
                st.save_checkpoint(work, it + 1, alive, sens,
                                   phi=getattr(opt, "phi", None),
                                   x=getattr(opt, "x", None),
                                   ctrl=(ctrl.state() if ctrl else None),
                                   weights=(wctrl.state() if wctrl else None))
                log("[oropt] CONVERGED")
                break

            # ---- next design ----------------------------------------------
            sens_prev = sens
            # Multipoint mode: record this iteration's measured point, then let
            # the controller pick the target from its fitted boundary (it falls
            # back to the optimiser's classic gate until the fit is usable).
            if ctrl is not None:
                ctrl.record(vf, violation)
                target_vf = ctrl.next_target_vf(vf, feasible,
                                                violation=violation)
            else:
                target_vf = opt.next_target_vf(vf, feasible,
                                               violation=violation)
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
            alive_before = alive
            alive = opt.update(alive, update_sens, target_vf)
            # Manufacturing constraints (min/max member size, symmetry, casting,
            # extrusion, overhang) on the fresh alive mask; re-drop islands a
            # constraint may have created. No-op unless configured. The unbiased
            # sensitivity guides the max-member carve toward the least-useful
            # material. The pruned mask is what the next opt.update() receives,
            # so a field-carrying optimiser (level-set) re-syncs to it there.
            if manufacturing_active(cfg.manufacturing):
                alive = apply_manufacturing(alive, mesh, cfg.manufacturing,
                                            protected, sensitivity=sens)
                alive = mesh.keep_connected(alive, anchor)
            if hold_void:
                # Hold the keep-out candidates void: neither the optimiser update
                # nor a manufacturing mirror may place material in the neighbour.
                alive = alive & keep_growable
            removed_now = int((alive_before & ~alive).sum())
            if _removal_spike(removed_now, removal_hist):
                log(f"[oropt] WARNING: this update removed {removed_now} "
                    f"elements, far above the recent per-iteration rate "
                    f"(last {len(removal_hist)}: {removal_hist}) -- likely a "
                    "wholesale low-energy collapse; inspect topology_latest.vtu "
                    "before trusting the next solve")
            removal_hist = (removal_hist + [removed_now])[-5:]
            # Post-update physics sanity audit (advisory: logs, never aborts).
            # Audits the mask about to be solved next against the one just solved.
            if getattr(cfg.run, "sanity_checks", True):
                rep = sanity_audit(mesh, alive, alive_before, anchor,
                                   min_layers=max(1, cfg.manufacturing.min_member_layers),
                                   protected=protected)
                for w in rep.warnings:
                    log(f"[oropt] SANITY: {w}")
            prev_vf, prev_target_vf = vf, target_vf
            st.save_checkpoint(work, it + 1, alive, sens_prev,
                               phi=getattr(opt, "phi", None),
                               x=getattr(opt, "x", None),
                               ctrl=(ctrl.state() if ctrl else None),
                               weights=(wctrl.state() if wctrl else None))
        else:
            status.state = "converged" if status.state == "running" else status.state
            status.message = status.message or "reached max_iter"
    finally:
        if status.state == "running":
            status.state = "stopped"
        st.write_status(work, status)
        # Post-run: best-effort OpenRadioss anim -> LS-Dyna d3plot of the **last
        # feasible** design, for EVERY load case (each in its own solve dir). Near
        # convergence the optimiser oscillates across the constraint boundary, so
        # the last *solved* iteration is often infeasible; the last feasible
        # iteration's animation is read from its per-iteration archive
        # (work/iter_NNNN/) so the d3plot matches the design the report renders,
        # falling back to the live solve dir when that archive isn't available.
        # Done while the run still owns the pid (so the GUI stays 'running' and
        # won't recycle solve/ mid-conversion); never let post-processing affect
        # the run's result/state.
        feas_it = reported_iteration(work)
        for i, case in enumerate(cases):
            try:
                solve_dir = _case_solve_dir(solve_root, n_cases, i)
                # Pass the case's own stem so convert_final finds its <stem>A0*
                # animation. Distinct stems -> the per-case d3plot files never
                # collide in work/d3plot/.
                anim_dir, used_archive = _final_anim_dir(
                    work, feas_it, case.stem, n_cases, solve_dir)
                if used_archive:
                    log(f"[oropt] d3plot: {case.stem}: converting the last feasible "
                        f"design (iteration {feas_it})")
                else:
                    log(f"[oropt] d3plot: {case.stem}: converting the last solved "
                        f"iteration (no archived feasible-iteration animation)")
                convert_final(cfg, anim_dir, work, stem=case.stem, log=log)
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
        # Independent manufacturability audit of the FINAL design (oropt.mfg_verify):
        # re-measures the shipped mask against the configured manufacturing limits
        # from raw geometry — NOT a re-run of the enforcement code — so a bug in a
        # constraint, or a violation the post-manufacturing keep_connected
        # reintroduced, is caught. Written to manufacturability.json for the report;
        # best-effort, never affects the run. Skipped when no constraint is active.
        try:
            if manufacturing_active(cfg.manufacturing):
                rep = mfg_verify(mesh, alive, cfg.manufacturing, protected=protected)
                write_manufacturability(work, rep)
                verdict = "PASS" if rep.ok else "FAIL"
                log(f"[oropt] manufacturability audit: {verdict} "
                    f"({len(rep.checks)} checks)")
                for w in rep.warnings:
                    log(f"[oropt] manufacturability: {w}")
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] manufacturability: unexpected error during audit: {exc}")
        # Automatic post-run summary (report.html/report.md) from the status &
        # history this run wrote. Read-only and best-effort; never affects the run.
        try:
            write_report(cfg, work, log)
        except Exception as exc:  # noqa: BLE001
            log(f"[oropt] report: unexpected error during report: {exc}")
        st.clear_pid(work)
    return status


def write_manufacturability(work: Path, report) -> None:
    """Persist the final-design manufacturability audit (oropt.mfg_verify) to
    ``manufacturability.json`` in the run folder, for the report / a later review.
    The report dataclasses are plain, so ``dataclasses.asdict`` round-trips them.
    """
    import json
    payload = dataclasses.asdict(report)
    (Path(work) / "manufacturability.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")


def _scatter(results, elem_ids: np.ndarray) -> np.ndarray:
    """von-Mises mapped onto the full element array (0 where no result)."""
    out = np.zeros(elem_ids.size, dtype=float)
    pos, valid = id_positions(elem_ids, results.element_ids)
    out[pos] = results.vonmises[valid]
    return out


def _scatter_max(results_list, elem_ids: np.ndarray) -> np.ndarray:
    """Per-element worst (max) von-Mises across load cases, on the full element
    array. With one case this is exactly :func:`_scatter` of that case."""
    out = _scatter(results_list[0], elem_ids)
    for results in results_list[1:]:
        out = np.maximum(out, _scatter(results, elem_ids))
    return out

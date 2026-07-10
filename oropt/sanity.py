"""Pre-solve geometric sanity checks on the evolving alive-element set.

These are **pure, hermetic, geometry-only** checks run on the ``alive`` boolean
mask each iteration *before* the expensive FEA solve, to catch physically
nonsensical designs the optimisers can produce and hand to the solver. They read
numpy arrays / a :class:`~oropt.mesh.Mesh` (never a ``Config``), keep no global
state, use only numpy/scipy, and are deterministic.

Motivation — the level-set collapse of 2026-07-05/06
(``docs/levelset_stuck_analysis.md``): a run reached convergence-like behaviour
while quietly *destroying* the design — its final update removed ~38k elements,
99 % of them the lowest-energy quartile, in a few large connected chunks of thin
webs (~1.4 mm walls) 35-60 mm from any load path. The loop happily solved
severed thin webs and plateau-collapse shrapnel because nothing looked at the
*geometry* of the design before spending 2.5 hours solving it. `validate.py`
gates the *config*; this module extends that "gate before you pay" philosophy to
the per-iteration *physics/geometry* of the mask.

The three primitive checks, and the failure each guards against:

* :func:`disconnected_elements` — alive material not reachable (through the
  shared-node element graph) from the anchor set (BC/load/protected elements).
  Such material carries no load path; it is a severed island the solver would
  treat as floating (rigid-body / singular-stiffness territory). Reuses exactly
  the connectivity machinery of :meth:`oropt.mesh.Mesh.keep_connected`.
* :func:`thin_members` — load-bearing walls thinner than ``min_layers``
  element-adjacency hops, measured with a morphological *open* (erode then
  dilate over shared-node adjacency, the same operator
  :func:`oropt.manufacturing._morph_open` enforces, used here purely as a
  *measurement*). These are the 0.8-1.4 mm severable webs of the post-mortem.
* :func:`newly_exposed_faces` — internal tet faces that become free surface this
  step because the element on the other side was just deleted. This is the
  self-contact concern (two freshly exposed surfaces can interpenetrate under
  load), plus a cheap heuristic for freshly-punched interior cavities.

:func:`audit` aggregates the three into a :class:`SanityReport` with
human-readable ``warnings`` and an ``ok`` flag. Severity is **advisory by
default**: only a severe condition (disconnected elements, on by default) clears
``ok``; the caller decides whether to abort, back off, or merely log. All
thresholds are arguments with sensible defaults — nothing here reads config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .mesh import Mesh

#: The four triangular faces of a TETRA4, as index triples into its 4 nodes.
#: Every face of the mesh is generated once from these; a face shared by two
#: elements (same sorted node triple) is an interior face.
_TET4_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64)


# --------------------------------------------------------------------------- #
# 1. connectivity — severed islands
# --------------------------------------------------------------------------- #
def disconnected_elements(
    mesh: Mesh, alive: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, int]:
    """Alive elements not connected to the *anchor* set through shared nodes.

    *anchor* is a boolean element mask of the load-path anchors (the BC / load /
    protected elements — exactly the ``anchor`` the loop passes to
    :meth:`oropt.mesh.Mesh.keep_connected`). An alive element is *orphaned* when
    no shared-node path reaches an alive anchor element: it is a floating island
    severed from every load path (the failure mode the post-mortem's thin-web
    collapse produced). Connectivity is computed with the very machinery the loop
    itself uses to drop islands — the shared-node element graph and
    ``scipy.sparse.csgraph.connected_components`` inside ``keep_connected`` — so
    "disconnected" here means exactly what the loop's own prune means.

    Returns ``(orphan_mask, count)``: the boolean mask of alive-but-orphaned
    elements (aligned with the element arrays) and its popcount. When no anchor
    element is alive, *every* alive element is reported orphaned.
    """
    alive = np.asarray(alive, dtype=bool)
    if anchor is None:
        return np.zeros(alive.shape, dtype=bool), 0
    anchor = np.asarray(anchor, dtype=bool)
    kept = mesh.keep_connected(alive, anchor)
    orphan = alive & ~kept
    return orphan, int(orphan.sum())


# --------------------------------------------------------------------------- #
# 2. thin members — severable webs
# --------------------------------------------------------------------------- #
def _element_adjacency(mesh: Mesh) -> sparse.csr_matrix:
    """Symmetric, reflexive shared-node element adjacency (``A[i, j] > 0`` iff
    elements *i* and *j* share a node).

    A local copy of :func:`oropt.manufacturing._element_adjacency`: the same
    incidence product the rest of the mesh code uses, so "neighbour" means what
    ``keep_connected`` / ``protected_mask`` mean by it. Kept local so a *measure*
    never reaches into the manufacturing *enforcement* module.
    """
    inc = mesh._incidence(np.arange(mesh.n_elements)).astype(np.float64)
    return (inc @ inc.T).tocsr()


def _erode(a: np.ndarray, adj: sparse.csr_matrix, layers: int) -> np.ndarray:
    """``layers`` morphological erosions: turn off any alive element that has a
    void (dead) neighbour. Anti-extensive."""
    out = a.copy()
    for _ in range(layers):
        dead = ~out
        has_dead_neighbour = adj.dot(dead.astype(np.float64)) > 0.0
        out = out & ~has_dead_neighbour
    return out


def _dilate(a: np.ndarray, adj: sparse.csr_matrix, layers: int) -> np.ndarray:
    """``layers`` morphological dilations: turn on any void element with an alive
    neighbour. Extensive."""
    out = a.copy()
    for _ in range(layers):
        out = out | (adj.dot(out.astype(np.float64)) > 0.0)
    return out


def thin_members(
    mesh: Mesh,
    alive: np.ndarray,
    min_layers: int,
    protected: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Flag alive elements lying in features thinner than *min_layers* hops.

    Measured with a morphological *open* over shared-node element adjacency —
    ``min_layers`` erosions then ``min_layers`` dilations, clipped to the alive
    set — the same operator :func:`oropt.manufacturing._morph_open` enforces, but
    used here only to *measure*: an alive element survives the open iff its
    feature contains an element at least ``min_layers`` hops from any void (i.e.
    the feature is at least ``min_layers`` elements thick). Everything the open
    removes is a sub-``min_layers``-thick feature — a thin web of the kind that
    got severed 35-60 mm from the load path in the level-set post-mortem. Pure
    erosion alone would also flag the *skin* of thick bulk; the dilation restores
    that skin, so only genuinely thin features remain flagged.

    When ``min_layers <= 0`` nothing is flagged (the check is off). If
    *protected* (BC/load/keep-out) is given, protected elements are excluded from
    the flagged set — an intentionally thin protected region (e.g. a load patch)
    is not a defect. Protected elements are *not* used to shield their
    neighbours; this is a geometric measurement, not the enforcement pass.

    Returns ``(thin_mask, count)``.
    """
    alive = np.asarray(alive, dtype=bool)
    if int(min_layers) <= 0:
        return np.zeros(alive.shape, dtype=bool), 0
    adj = _element_adjacency(mesh)
    opened = _dilate(_erode(alive, adj, int(min_layers)), adj, int(min_layers)) & alive
    thin = alive & ~opened
    if protected is not None:
        thin = thin & ~np.asarray(protected, dtype=bool)
    return thin, int(thin.sum())


# --------------------------------------------------------------------------- #
# 3. newly exposed faces — self-contact / cavities
# --------------------------------------------------------------------------- #
def _internal_face_pairs(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two element indices on each interior face of a tet mesh.

    Builds all ``4 * N`` triangular faces once (sorted node triples), groups
    identical triples, and returns ``(elem_a, elem_b)`` for every face shared by
    exactly two elements — the interior faces. Boundary faces (one owner) and any
    non-manifold face (>2 owners, not expected in a conforming mesh) are dropped.
    Fully vectorised; ``N`` is millions in production.
    """
    tets = np.asarray(tets, dtype=np.int64)
    n = tets.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    faces = tets[:, _TET4_FACES].reshape(-1, 3)          # (4N, 3)
    faces = np.sort(faces, axis=1)
    elem = np.repeat(np.arange(n, dtype=np.int64), 4)    # owner element per face
    # group identical faces; sort keys lexicographically so shared faces are adjacent
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    faces_s = faces[order]
    elem_s = elem[order]
    _, start, counts = np.unique(faces_s, axis=0, return_index=True, return_counts=True)
    interior = start[counts == 2]
    return elem_s[interior], elem_s[interior + 1]


def newly_exposed_faces(
    mesh: Mesh,
    alive: np.ndarray,
    prev_alive: np.ndarray,
    tets: np.ndarray | None = None,
) -> tuple[int, int]:
    """Count interior faces that become free surface this step, plus a cavity heuristic.

    A face shared by two elements is *newly exposed* when one element is still
    alive and the other was alive last step but was deleted this step
    (``prev_alive & ~alive``): the survivor gains a fresh free surface where solid
    used to back it. A large crop of these is the self-contact concern — two
    newly bared surfaces can interpenetrate under load, which the linear/implicit
    solve does not police.

    The tet node-connectivity is taken from ``mesh.conn_rows`` (the ``(N, 4)``
    node-row connectivity :class:`~oropt.mesh.Mesh` already stores) unless an
    explicit *tets* ``(N, 4)`` int array is passed — pass ``deck.elem_conn`` if
    you want raw node ids; either labelling gives identical face sharing.

    Returns ``(n_newly_exposed, n_cavity_faces)``. The second value is a
    **heuristic lower bound** on interior cavities: it counts the faces of
    deleted elements that are *fully enclosed* — all four faces interior with a
    currently-alive element on the far side — i.e. a single element carved out of
    solid bulk, the smallest closed self-contact pocket. It does **not** perform
    full topological cavity detection: a multi-element interior void is not
    counted (only its constituent fully-enclosed elements, if any), and a cavity
    that touches the mesh boundary is not closed and so not counted. Treat it as a
    cheap "did we just punch a sealed hole?" signal, not a proof.
    """
    alive = np.asarray(alive, dtype=bool)
    if prev_alive is None:
        return 0, 0
    prev_alive = np.asarray(prev_alive, dtype=bool)
    conn = mesh.conn_rows if tets is None else np.asarray(tets, dtype=np.int64)
    ea, eb = _internal_face_pairs(conn)
    if ea.size == 0:
        return 0, 0

    deleted = prev_alive & ~alive
    # newly exposed: survivor on one side, freshly deleted neighbour on the other
    newly = (alive[ea] & deleted[eb]) | (alive[eb] & deleted[ea])
    n_newly = int(newly.sum())

    # cavity heuristic: a deleted element all of whose 4 faces are interior and
    # backed by a currently-alive neighbour is a sealed single-element pocket.
    n = alive.shape[0]
    both = np.concatenate([ea, eb])
    interior_faces_per_elem = np.bincount(both, minlength=n)
    backed = np.concatenate([ea[alive[eb]], eb[alive[ea]]])
    alive_backed_per_elem = np.bincount(backed, minlength=n)
    enclosed = deleted & (interior_faces_per_elem == 4) & (alive_backed_per_elem == 4)
    n_cavity_faces = int(4 * int(enclosed.sum()))
    return n_newly, n_cavity_faces


# --------------------------------------------------------------------------- #
# 4. aggregate report
# --------------------------------------------------------------------------- #
@dataclass
class SanityReport:
    """Structured result of :func:`audit`.

    ``ok`` is ``True`` unless a *severe* condition fired (by default: any
    disconnected element). Every check's result is always populated so the caller
    can log/branch on the raw counts regardless of ``ok``. ``warnings`` holds the
    human-readable messages (empty on a clean design).
    """
    ok: bool
    disconnected_mask: np.ndarray
    n_disconnected: int
    thin_mask: np.ndarray
    n_thin: int
    n_newly_exposed_faces: int
    n_cavity_faces: int
    warnings: list[str] = field(default_factory=list)


def audit(
    mesh: Mesh,
    alive: np.ndarray,
    prev_alive: np.ndarray | None,
    anchor: np.ndarray | None,
    *,
    min_layers: int = 0,
    tets: np.ndarray | None = None,
    protected: np.ndarray | None = None,
    severe_disconnected: bool = True,
    max_exposed_faces: int | None = None,
) -> SanityReport:
    """Run all geometric sanity checks on *alive* and aggregate a :class:`SanityReport`.

    How the loop should call this (all values it already has to hand, *before* the
    solve):

    * ``mesh``        — the design :class:`~oropt.mesh.Mesh`.
    * ``alive``       — the candidate mask about to be solved (post manufacturing
      + ``keep_connected``).
    * ``prev_alive``  — the previous iteration's alive mask (``alive_before`` in
      the loop). Pass ``None`` on iteration 0 to skip the face check.
    * ``anchor``      — the connectivity anchor the loop built (the BC/load
      ``protected_mask`` it passes to ``keep_connected``). Pass ``None`` to skip
      the connectivity check.
    * ``min_layers``  — thin-member threshold in adjacency hops (default 0 = off).
      A natural choice is the run's ``manufacturing.min_member_layers``, or a
      small fixed guard like 1-2.
    * ``tets``        — leave ``None`` to use ``mesh.conn_rows``; pass
      ``deck.elem_conn`` only if you specifically want raw node-id labelling.
    * ``protected``   — optional protected mask; protected elements are excluded
      from thin-member flags.

    Severity is advisory: only ``severe_disconnected`` (default on) clears
    ``ok`` — every other finding is a warning the caller may act on or ignore.
    ``max_exposed_faces`` (default ``None`` = never) adds an advisory warning when
    the newly-exposed-face count exceeds it; a cavity finding always warns.
    Nothing here aborts the run — the caller decides.
    """
    warnings: list[str] = []

    disc_mask, n_disc = disconnected_elements(mesh, alive, anchor)
    if n_disc > 0:
        sev = "SEVERE" if severe_disconnected else "warning"
        warnings.append(
            f"[{sev}] {n_disc} alive element(s) are disconnected from the "
            "anchor (BC/load) set -- severed from every load path"
        )

    thin_mask, n_thin = thin_members(mesh, alive, min_layers, protected)
    if n_thin > 0:
        warnings.append(
            f"[warning] {n_thin} alive element(s) lie in features thinner than "
            f"{min_layers} adjacency hop(s) -- thin/severable webs"
        )

    n_exposed, n_cavity = newly_exposed_faces(mesh, alive, prev_alive, tets)
    if max_exposed_faces is not None and n_exposed > max_exposed_faces:
        warnings.append(
            f"[warning] {n_exposed} internal face(s) newly exposed as free "
            f"surface this step (> {max_exposed_faces}) -- possible self-contact"
        )
    if n_cavity > 0:
        warnings.append(
            f"[warning] {n_cavity} face(s) bound freshly sealed interior "
            "cavities (single-element pockets) -- self-contact risk"
        )

    ok = not (severe_disconnected and n_disc > 0)
    return SanityReport(
        ok=ok,
        disconnected_mask=disc_mask,
        n_disconnected=n_disc,
        thin_mask=thin_mask,
        n_thin=n_thin,
        n_newly_exposed_faces=n_exposed,
        n_cavity_faces=n_cavity,
        warnings=warnings,
    )

# Research spike: user-defined "growth boxes" — letting the optimiser ADD material where the original part had none

**Verdict: feasible, and cheaper than it sounds — the optimisers already know how to
add material; what's missing is *candidate elements to add* and a way to mark them.**
The recommended design is a config-defined list of boxes (mirroring LS-DYNA
[`*DEFINE_BOX`](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/define_box_lsdyna_r.htm)
/ Radioss `/BOX/RECTA`: two corner points, multiple boxes allowed) over a
**pre-meshed expansion volume**: every design element whose centroid lies inside a
box starts **void** instead of alive, and the existing bi-directional machinery
(BESO add-back, TOBS `dx ∈ {0,+1}` flips, level-set φ growth) grows material into
it when the load path wants it. No solver-side blockers exist. What oropt must
*not* attempt (in phase 1) is generating mesh itself. *(Phase 2 — since
implemented, `oropt.growthmesh` — does generate it, but **node-conformally**,
so nothing about the run changes; see §2 and §5.)*

---

## 1. Why this is compatible with oropt's architecture

oropt does hard-kill element deletion on a fixed TET4 universe: the deck's single
`/TETRA4/<design_part_id>` block is parsed once (`deck.py`), and each iteration a
boolean `alive` mask over exactly those `elem_ids` is re-written verbatim
(`Deck.write`). Three facts make "adding material" natural rather than alien:

1. **All three optimisers are already bi-directional.**
   * BESO: dead elements get raw sensitivity 0 but are pulled up by the spatial
     filter (`map_sensitivity` doc says exactly this: *"dead/absent get 0 and are
     pulled up only by the filter, which is what makes them eligible for
     bi-directional add-back"*); `Beso.update` ranks **all** removable elements —
     dead ones included — against the volume budget, capped by `max_add_ratio`.
   * TOBS: a currently-void element's flip bound is `dx ∈ {0,+1}`; the back-off
     branch of `_volume_band` explicitly picks the K highest-sensitivity *voids*.
   * Level-set: `_init_phi` builds φ from **whatever alive mask it first sees**
     (not from "all ones"), and interface voids acquire positive velocity from the
     filtered energy of their alive neighbours, so φ grows outward.

2. **The loop starts from `alive = np.ones(...)` by convention only**
   (`loop.py`, "initial / resumed state"). Nothing downstream assumes the initial
   mask is all-True: `Deck.write`, free-node pinning, `/SURF/PART/EXT` contact
   regeneration, snapshots, smoothing, animation, d3plot all operate on arbitrary
   masks. Changing the initial mask to "everything except in-box elements" is the
   entire core of the feature.

3. **Dead elements are already solver-safe.** Nodes referenced by no alive element
   are auto-pinned via the injected `/GRNOD/NODE` + `/BCS` (the free-node guard —
   proven at 16 352 pinned nodes with NORMAL TERMINATION). A box region that
   starts fully void is just a large pinned node set at iteration 0; as elements
   come alive their nodes are referenced again and drop out of the pinned set on
   the next write. Iteration 0 therefore solves **exactly the original part** —
   the validated baseline is unchanged.

This is also how industry treats the problem: commercial topology tools
(OptiStruct, Tosca, LS-TaSC) always optimise a user-meshed **design envelope**
that is larger than the expected part. The oropt twist — start the envelope
extension *void* and grow into it, instead of starting solid and carving — is an
established BESO-family variant (ESO/BESO literature has always allowed element
insertion; recent work such as PCM-BESO's "boundary growth" and *adaptive design
domain* topology optimisation formalises growing past the initial envelope).

## 2. The one hard prerequisite: candidate elements must exist in the deck

An element-deletion optimiser cannot create elements. "Add material where there
was none" therefore decomposes into two very different sub-problems:

| | Route A — user pre-meshes the expansion volume (phase 1) | Route B — auto-mesh + `/INTER/TYPE2` tie (**REJECTED**) | Route B′ — auto-mesh, node-conformal direct creation (**phase 2, implemented**) |
|---|---|---|---|
| Mesh source | User extends the design part with tet-meshed box volumes in the pre-processor (or in the source `.k` before `k_to_rad_converter`) | oropt voxelises each box into hex-split tets, subtracts the part overlap, injects `/NODE` + `/TETRA4` + tied-contact cards | `oropt.growthmesh` (explicit PREPARE step, CLI + GUI): TetGen `-Y` constrained tetrahedralisation of a PLC embedding the part's exterior surface; new `/NODE` + `/TETRA4` cards spliced into **extended starter decks** the user inspects before running |
| Interface to part | **Conformal** (shared nodes; imprint + node-equivalence at the part surface) | Non-conformal — needs `/INTER/TYPE2` tied contact injection | **Conformal by construction** — `-Y` preserves the part's surface facets/vertices exactly (Steiner points interior-only), so new tets reuse the part's surface *node ids* |
| `keep_connected` | Works as-is (shared nodes ⇒ grown material is connected) | Broken — tied interfaces share no nodes, so grown material is dropped as a floating island unless adjacency is artificially augmented | Works as-is (shared nodes, same as Route A) |
| Deck philosophy | Preserved: deck still edited verbatim, cards only omitted | Violated: oropt writes new geometry *and* contact cards, and the tied master surface would have to be maintained as elements are deleted | Preserved at run time: the PREPARE step writes a new starter once, up front; the run still edits it verbatim, byte-identical phase-1 behaviour |
| Solver risk | None new (same mechanics as today) | **Fatal, on verified grounds**: Radioss implicit supports `/INTER/TYPE2` only in its *kinematic* formulation — no penalty fallback (Altair *Implicit Features and Compatibility*) — and a kinematic tie **conflicts with the `/BCS` free-node pinning** oropt injects on exactly the void-candidate interface nodes (two kinematic conditions on the same DOFs); the tied master surface also evolves as interface elements are deleted | None new: iteration 0 solves exactly the original part; new elements start void; the existing free-node pinning covers their nodes; the phase-1 run-start guards double as self-checks of the generated mesh |
| Effort | Manual pre-processor work per load case | Large (dual-mesh bookkeeping, contact, months of validation) | One command (`python -m oropt.growthmesh --config …`) or one GUI button |

Route A got the full user value first — material growing outside the original
part, multiple regions, any optimiser — at a fraction of the risk, and matches
the project's "the deck is edited verbatim" principle. Route B is **rejected
outright** (not merely deferred): the `/INTER/TYPE2` implicit
kinematic-formulation restriction and its clash with the free-node `/BCS`
pinning are structural, not an effort problem. Route B′ (phase 2, §5) removes
the pre-meshing burden while keeping every phase-1 mechanism untouched, because
the generated mesh is exactly what a careful user would have produced by hand:
node-conformal TET4 in the same `/TETRA4/<design_part_id>` block.

### Deck-preparation checklist (Route A, user-facing docs material)

*(Everything below is what `oropt.growthmesh` automates — it allocates ids
above max(existing) and ≥ `design_node_min`, splices the same cards into every
load case's first `/NODE` block and design `/TETRA4` block, and re-runs the
run-start guards on the result before writing. The checklist stays relevant
for hand-meshed regions.)*

* Mesh each box volume with TET4 and merge it into the **same design part**, so
  the cards land in the single `/TETRA4/<design_part_id>` block that
  `Deck._find_elem_block` parses (multiple blocks with the same part id are *not*
  parsed today — either keep one block or extend `deck.py` to concatenate them).
* New nodes must go into the **first `/NODE` block** (`_find_node_block` reads
  only that one) and must have ids **≥ `design_node_min`** — otherwise the
  free-node guard will not pin them while void and the implicit tangent goes
  singular. This must be a hard validation error.
* The interface must be **node-conformal** (imprint the part surface mesh, then
  equivalence coincident nodes). Without shared nodes the grown material is
  structurally disconnected *and* `keep_connected` deletes it immediately.
* Contact: `/SURF/PART/EXT` masters regenerate from surviving elements — free.
  But any contact **slave node group** that is a fixed node list (e.g. the
  self-contact `/INTER/TYPE7` slaves) will not include the new nodes unless the
  user adds them; worth a docs note.
* Multi-load-case runs: **every** case's deck must be rebuilt with the identical
  extended mesh (the loop already hard-errors when `elem_ids` differ).
* Same part ⇒ same property/material: grown material is automatically the same
  AlSi10Mg as the part. If a different material were ever wanted, that's a
  separate part id and out of scope.

## 3. Proposed design

### 3.1 Config schema (mirrors `*DEFINE_BOX` / `/BOX/RECTA`)

`*DEFINE_BOX` is: a box id + two opposite corner points, optionally in a local
skew system (`LOCAL` variant → `/BOX/RECTA` + `/SKEW/FIX`). Phase 1 supports the
global axis-aligned form, as a list (multiple boxes = union), placed under
`model:` since it is design-domain geometry:

```yaml
model:
  growth_boxes:                    # regions of the pre-meshed expansion volume
    - {name: rib_top,  x_min: 10.0, x_max: 40.0, y_min: -5.0, y_max: 5.0, z_min: 0.0, z_max: 25.0}
    - {name: gusset_l, x_min: -20.0, x_max: 0.0,  y_min: -5.0, y_max: 5.0, z_min: 0.0, z_max: 12.0}
```

* New `GrowthBox` dataclass in `config.py` (name + 6 floats), coerced from dict
  rows in `Model.__post_init__` exactly like `AnimateOpts.custom_views` coerces
  `CustomView`, so YAML round-trips and GUI table rows both work.
* `unknown_keys` gets a `_list_of("model.growth_boxes", ...)` clause so typo'd
  fields are flagged.
* Extensible later: a `shape:` field (`box`/`cylinder`/`sphere`, matching
  Radioss `/BOX/CYLIN`, `/BOX/SPHER`) and a local-skew variant — membership tests
  are one-liners on centroids.
* Optional phase-1.5 alternative input: parse `/BOX/RECTA` cards straight from
  the deck (a `Deck.group_nodes`-style helper), so boxes authored in the
  pre-processor travel with the model. Config boxes remain the primary,
  GUI-editable path.

**Semantics** (must be documented loudly, because it *differs* from what
DEFINE_BOX does in a crash deck): a growth box does not "select entities for an
operation" — it declares *this part of the design mesh is candidate material
that starts void*. Membership = element **centroid inside the box** (inclusive
bounds; centroids already exist on `Mesh`). Union over all boxes. A box that
overlaps the original part volume will void those elements at start too — legal
(deliberate carve-and-regrow) but surprising, so the run log should state how
many initially-alive-looking elements each box voided.

### 3.2 Loop changes (the whole feature is ~10 lines here)

```python
candidate = mesh.in_boxes_mask(m.growth_boxes)        # new Mesh helper, vectorised centroid test
alive = ~candidate                                     # instead of np.ones(...)
protected &= ~candidate                                # see 3.3
```

plus logging (`N candidate elements in M growth boxes, starting void`) and two
startup guards:

* **Empty box** → the user defined a box but forgot to extend the mesh: zero
  candidate elements inside it. This must be a loud warning (arguably an error) —
  otherwise the feature silently does nothing. It cannot live in `validate.py`
  (config-only, no deck parse); it belongs at loop start and, on demand, behind a
  GUI "preview boxes" button.
* **Disconnected candidate region** → candidate elements sharing no node with any
  initially-alive element can never survive `keep_connected`; error out with the
  box name (this is exactly the "forgot to equivalence the interface nodes"
  failure).

`validate.py` additionally checks pure config sanity: `min < max` per axis,
duplicate/empty names.

### 3.3 Interaction with the protected set

`mesh.protected_mask` seeds from node groups and dilates `protect_layers` hops —
a box adjacent to the BC region would make some *void* candidates "protected",
and every optimiser force-protects (`cand = protected.copy()` in BESO,
`| self.protected` in the level-set, `new_alive |= prot` in TOBS): those box
elements would **materialise instantly at iteration 1** regardless of
sensitivity. Candidates are new material with no BC/contact/keep-out claims, so
the right rule is `protected &= ~candidate` (with a log line when the
intersection was non-empty). The anchor mask needs no change — dead anchor
elements are ignored by `keep_connected` (`seed_mask[alive_idx]`).

### 3.4 Volume bookkeeping: keep `V0` = part + boxes

Every optimiser's `V0 = mesh.volumes.sum()` becomes the **envelope** volume, so:

* Initial vf = V_part / V0 < 1 (today it is exactly 1). `target_volume_fraction`
  is now a fraction *of the envelope* — a semantics change users must be told
  about when they add boxes (a run log line: `vf(start)=0.71 of the enlarged
  design space` covers it).
* This keeps `next_target_vf`, the BESO budget maths, TOBS's `dV`, and the
  level-set bisection all working **unmodified** — the alternative (V0 = part
  only, vf can exceed 1) breaks the `min(1.0, ...)` back-off clamp and the
  convergence gate for zero benefit.
* A pleasant emergent feature: setting `target_volume_fraction` *above* the
  initial part fraction turns the run into **grow-to-target** —
  `next_target_vf = max(tvf, vf·(1−ER))` immediately targets growth and the
  add-back mechanisms fill the best-ranked box material. Worth a test and one
  README sentence.

### 3.5 How each optimiser behaves with boxes (no algorithm changes needed)

* **BESO** — two growth channels, both already coded:
  *Relocation while feasible*: interface voids in a box next to a hot part
  surface get large filtered sensitivity and out-rank low-energy alive elements
  inside the same shrinking budget — material migrates into the box while the
  total still shrinks. *Reinforcement when infeasible*: the back-off raises the
  target by ER and the threshold admits the highest-ranked voids — with boxes,
  "add material back" can now mean "add it *outside the original envelope*",
  which is precisely the user's ask.
  Pacing caveats: growth advances at most ~one `filter_radius` per iteration
  (voids farther than the filter radius from alive material have identically
  zero sensitivity — nucleation deep inside a box is impossible, by design), and
  `max_add_ratio` (default 0.01) halves the reachable back-off step
  (ER = 0.02) — expect to recommend raising it to ≈ ER for box runs; a
  validation warning when `max_add_ratio < evolution_rate` and boxes exist is
  cheap.
* **TOBS** — void candidates are ordinary `{0,+1}` variables; the ILP simply
  gets (say) 5–15 % more binaries, and the ε-relaxed volume band plus
  `flip_limit` already bound the step. No change.
* **Level-set** — φ initialises negative in the boxes, interface velocity grows
  it outward, the bisected uniform shift handles the volume target; smoother
  grown geometry than BESO's, likely the nicest-looking ribs. No change.
* **Manufacturing constraints** — applied to the alive mask after the update, so
  min-member/symmetry/overhang automatically govern grown material too (a grown
  rib must be printable). Zero change; genuinely good synergy.

### 3.6 Reporting / GUI / persistence touches

* `status.py`: add `elements_candidate` and `elements_grown` (alive ∩ candidate)
  counters; Monitor shows "grown into boxes: N elements / X mm³".
* `report.py`: "% mass removed" can now legitimately be negative (net
  reinforcement run) — wording tweak, and state initial vf < 1 explicitly.
* `animate.py`: already frames on the union of all snapshots' bounds, so growth
  beyond the initial silhouette stays in frame. No change.
* GUI (`gui/app.py`): a growth-box table editor (name + 6 coords per row,
  add/remove) alongside the keep-out fields; box count/summary in the sidebar.
  Per the established pattern, Start/queue snapshots the config after tabs
  render, so box edits need no special handling. Nice-to-have: wireframe box
  overlay in the Monitor's 3D view (`_render.py` draws pyvista `Box` outlines) —
  placing coordinates blind is the worst part of the UX otherwise.
* Resume: checkpointed `alive` masks are aligned to a *mesh*; resuming an old
  run against a newly extended deck mismatches shapes. Cheap guard: store
  `elements_total` (and ideally a hash of `elem_ids`) in the checkpoint and
  refuse resume on mismatch with a clear message.

## 4. Risks and open questions

1. **Pre-meshing burden (the real cost of phase 1).** The user must produce a
   conformal extended mesh per load case and re-convert. This is standard
   pre-processor work (imprint + equivalence), but it is manual — which is
   exactly what the phase-2 PREPARE step (`oropt.growthmesh`, §5) now
   automates, node-conformally and without the tied-interface solver risk.
2. **Growth starvation.** Filter-radius-limited advance means thick boxes fill
   slowly (~1 element layer/iter) and only while the interface stays
   high-energy. Mitigations: larger `filter_radius` for box runs, higher
   `max_add_ratio`, grow-to-target (§3.4). Should be studied on the coarse proxy
   mesh before the 575k model.
3. **Oscillation.** Add-back next to removal can ping-pong (`history_weight`
   damping exists); box interfaces add surface area where this can happen. The
   convergence window already guards the stop criterion; watch the vf trace in
   early runs.
4. **Baseline drift when the part itself is re-meshed.** If the user re-meshes
   the *union* instead of imprinting onto the existing part mesh, iteration-0
   results shift slightly from the validated baseline (σ_max = 308.305 MPa
   reference). Prefer the imprint route in docs.
5. **Convergence wording.** `_converged` treats `vf ≤ target` as satisfied; a
   grow-to-target run that stalls below target can report "converged at target
   volume" while short of it. Cosmetic; adjust the message when boxes are active.
6. **`elements_total` meaning shifts** in status/GUI (now includes never-built
   candidates). Display "alive / part + candidates" split to avoid confusion.

## 5. Suggested phasing & effort

* **Phase 1 (small, high value):** `GrowthBox` config + `unknown_keys` +
  validation; `Mesh.in_boxes_mask`; loop initial-mask/protected/guard changes;
  status counters; report wording; GUI table; hermetic tests (initial mask,
  growth-into-box via each optimiser's update on a toy mesh, protected
  exclusion, disconnected-candidate guard, empty-box warning, YAML round-trip,
  resume-shape guard). Everything is testable without OpenRadioss, matching the
  existing 124-test hermetic suite. Rough size: a few hundred lines including
  tests — comparable to the stress-exclusion feature.
* **Phase 1.5 (nice-to-have) — implemented:**
  * **Shapes** — `GrowthBox.shape` is `box` (default) / `cylinder` / `sphere`
    (mirroring `/BOX/RECTA` / `/BOX/CYLIN` / `/BOX/SPHER`). Sphere = centre +
    radius; cylinder = two axis end-points + radius (finite, capped). The six
    box bounds are unchanged, so existing box YAMLs parse byte-identically. The
    centroid-in-primitive tests are one-liners in `mesh.primitive_member`; per-shape
    required fields are validated in `validate.py`; the GUI table gained a `shape`
    column with nullable per-shape coordinate columns (`gui/boxes.py` row helpers
    stay Streamlit-free + unit-tested).
  * **Polyhedron regions** (added after phase 2) — `shape: polyhedron` defines a
    region by an **arbitrary explicit node set** (`points: [[x, y, z], ...]`,
    ≥ 4 nodes, every coordinate given — no defaults, no inference); membership is
    centroid inside the **convex hull** of the points
    (`scipy.spatial.Delaunay(points).find_simplex(centroids) >= 0`), so an
    arbitrary warped 8-node brick is the convex case and a non-convex point set
    is treated as its hull. Coplanar/duplicate points (zero-volume hull) are a
    validation error. The overlay draws the hull's edges
    (`scipy.spatial.ConvexHull`); `growthmesh.region_bounds` uses `points.min/max`
    so the phase-2 generator works unchanged. In the GUI the node list gets its
    own name-keyed table (one x/y/z row per node, dynamic rows — the oriented-frame
    pattern), since N points don't fit the fixed-column region table.
  * **Oriented (local-system) boxes** — an optional local frame
    (`origin` + `x_axis` + `xy_axis`, Gram-Schmidt-orthonormalised in
    `mesh.local_frame_basis`) mirrors `*DEFINE_BOX_LOCAL` -> `/BOX/RECTA` +
    `/SKEW/FIX`: centroids are transformed into the frame before the bounds test.
  * **3D overlay** — `mesh.overlay_primitives` emits wireframe-outline descriptors
    (box edges / sphere / cylinder) drawn over the topology as red wireframe actors
    in the GUI Monitor's 3D view *and* the report's render (the report render still
    runs in the isolated off-screen subprocess, so the overlay never risks the
    headless-CI segfault). So coordinates can be placed visually instead of blind.
  * **`/BOX/RECTA` deck input** — `Deck.box_recta` parses a `/BOX/RECTA/<id>` card
    (a `group_nodes`-style helper); a `GrowthBox.deck_box_id` references it and is
    resolved to concrete corners at run start (`loop.resolve_growth_boxes`), as an
    alternative to literal coordinates.
  * **GUI "preview regions" button** — on the Input tab, loads the primary case's
    starter deck in-process (pure Python, no VTK) and reports each region's
    centroid-in-region element count plus the run-start guard verdict
    (`loop.preview_growth_boxes`), so a mis-placed or un-meshed region is caught
    before a multi-hour run instead of when the loop aborts.
* **Phase 2 — implemented: auto-generated, node-conformal candidate mesh
  (`oropt.growthmesh`).** An explicit **PREPARE step** — `python -m
  oropt.growthmesh --config cfg.yaml` (`--size-factor`, `--min-ratio`,
  `--out-dir`, `--dry-run`) or the GUI's **⚙️ Generate growth mesh** button next
  to the region preview — never something hidden inside run start. The earlier
  route-B sketch (auto-mesh + `/INTER/TYPE2` tie) was **rejected on verified
  solver grounds** — Radioss implicit supports `/INTER/TYPE2` only in its
  kinematic formulation (no penalty fallback, per Altair *Implicit Features and
  Compatibility*), which conflicts with the `/BCS` free-node pinning oropt
  injects on exactly the void-candidate interface nodes, and the tied master
  surface would evolve as interface elements are deleted. Instead the generated
  mesh is **node-conformal with the part**, so every phase-1 mechanism
  (`keep_connected`, run-start guards, verbatim per-iteration rewrite,
  free-node pinning, `/SURF/PART/EXT` regeneration) works untouched. How:
  * **Surface** — the design part's watertight exterior surface is extracted as
    the TET4 boundary faces (faces used by exactly one element), keeping the
    original node ids per face vertex.
  * **PLC + TetGen** — that surface is embedded as internal facets in a domain
    box (the AABB of part ∪ regions, slightly inflated), tetrahedralised by the
    pip `tetgen` package (the pyvista-maintained wrapper; optional extra
    `oropt[growthmesh]`; **TetGen itself is AGPL** — evaluate before
    redistributing) with `-p -q -a -Y`: `-Y` preserves the input facets and
    vertices exactly (Steiner points interior-only), which is what guarantees
    the output shares the part's surface **nodes**. One TetGen subtlety, found
    empirically and locked in by a comment + test: under `-Y` interior
    refinement points that *encroach* an unsplittable boundary facet are
    rejected, so leaving the domain walls as giant triangles silently vetoes
    the `-a` sizing — the walls are therefore pre-subdivided at ~2× the target
    edge (`growthmesh.box_shell`), with the domain margin keeping regions clear
    of their encroachment zones.
  * **Classification** — output tets are classified by centroid: inside the
    part (exact point-in-tet against the existing elements) → duplicates,
    dropped; outside every region (the same `mesh.primitive_member` geometry as
    the candidate mask) → scaffolding, dropped. The survivors are the new
    candidate elements; preserved surface vertices are mapped back to original
    node ids by coordinate match, the rest get **new node ids above
    max(existing) and ≥ `design_node_min`** (so the free-node guard can pin
    them while void); element ids go above max(existing); node ordering is
    flipped to the deck's own orientation sign.
  * **Output contract** — the same new `/NODE` + `/TETRA4` lines are spliced
    into *every* load case's starter (`Deck.extended_lines`; all cases share
    the mesh) and written as a full inspectable deck set under
    **`<case_dir>/growth_mesh/`** (`<stem>_0000.rad` extended, `<stem>_0001.rad`
    engine copied verbatim). Point `model.case_dir` there — the CLI prints the
    line, the GUI button rewrites it — and run: the run itself is byte-identical
    phase-1 behaviour, iteration 0 still solves exactly the original part.
  * **Self-checks** — the phase-1 run-start guards (empty region, node-min,
    reachability) are re-run on the extended deck **before anything is
    written**, and generated-mesh stats (element/node counts, per-region
    counts, min/median shape quality, sizing) are reported. Element sizing
    follows the part's mean surface edge length × `size_factor`.
  * **Tests** — the pipeline (surface extraction, PLC assembly, classification,
    id allocation, splicing, multi-case consistency, guard integration) is
    hermetic against a fabricated backend; TetGen-backed end-to-end tests sit
    behind `pytest.importorskip("tetgen")`, and `tetgen` is in the dev extra so
    CI exercises them.
  * **Placement remains the user's job** — the generator only sees the design
    part; keep regions clear of *other* parts (rigid bodies, shells), or grown
    material may interpenetrate them.

## 6. References

* Altair Radioss LS-DYNA-input reference, [`*DEFINE_BOX`](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/define_box_lsdyna_r.htm) — box id + two corner points, global or `LOCAL` skew variant (maps to `/BOX/RECTA` + `/SKEW/FIX`).
* Altair Radioss user guide, *Implicit Features and Compatibility* — the implicit solver's supported-keyword matrix: `/INTER/TYPE2` is available in the **kinematic formulation only** (no penalty fallback), the basis for rejecting the tied-interface route (§2).
* [TetGen](https://wias-berlin.de/software/tetgen/) (Hang Si, WIAS Berlin; **AGPL-3.0**) via the [pyvista-maintained `tetgen` wrapper](https://github.com/pyvista/tetgen) — constrained Delaunay tetrahedralisation; the `-Y` switch preserves the input surface mesh exactly (phase 2's conformity guarantee).
* Sivapuram & Picelli, *Topology optimization of binary structures using ILP* (TOBS), FEAD 139 (2018) — void flip bounds `{0,+1}` (already implemented in `tobs.py`).
* Huang & Xie, *Evolutionary Topology Optimization of Continuum Structures* (2010) — bi-directional add-back via filter-extrapolated void sensitivities; insensitivity of converged topology to the initial design for compliance problems.
* [PCM-BESO (Optim. Eng. 2024)](https://link.springer.com/article/10.1007/s11081-024-09917-0) — boundary-growth technique expanding the design representation beyond the initial domain, AM-oriented.
* [Rong et al., *Structural topology optimization with an adaptive design domain*, CMAME (2021)](https://www.sciencedirect.com/science/article/abs/pii/S0045782521006435) — formalises growing the design domain during optimisation.
* `docs/simp_spike.md` — the sibling spike whose format this document follows.

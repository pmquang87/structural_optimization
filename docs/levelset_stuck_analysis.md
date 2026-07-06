# Why the elevator-linkage level-set run stalled (2026-07-05/06)

Root-cause analysis of the level-set run in
`implicit_6kN_elevator-linkage_neutral-pull__BC-A_Erpro-Wie-Gebaut_foxcore-rund\opti_run1\growth_mesh`
(config `queue_configs/elevator_linkage_dispfix.yaml`: optimizer `levelset`,
ER 0.015, target_vf 0.3, dt 1.0, smoothing_passes 3, band_width 3.0,
backoff_gain 1.0, backoff_cap 4.0, damping_threshold 0.9, filter_radius 1.0,
manufacturing `min_member_layers: 1`; 205,614 growth-box candidates; 960,843
protected elements = 42.3%; V0 = 104,332; the run executed `2d3c1ce`,
i.e. **pre-PR #57**).

> **Note on PR #57** (`fix/levelset-nucleation-backoff`, merged 2026-07-06
> 07:48Z, while this analysis was in progress): it implements two of the fixes
> this analysis calls for — the back-off floor and hole nucleation — but **not
> the primary leak**, which is still present on `master` (measured below).
> See "Status relative to PR #57".
>
> **Note on the fix** (`fix/levelset-volume-leak`, 2026-07-06, after this
> analysis): the three "still open" requirements below have landed —
> `LevelSet.update` now re-syncs phi to the incoming (post-prune) mask and
> refunds the pruned volume to the bisection budget, phi is checkpointed, and
> the loop warns on grow-stall / removal-spike signatures. Regression
> coverage: `tests/test_levelset_leak.py`. The two reproduction scripts below
> demonstrate the *defect* and are era-pinned to commit `2280f86`.

## TL;DR

The run was not merely stuck — it was **ratcheting the wrong way, and its
final update quietly destroyed the design**. From iteration 2 the design sat
0.6–1.7 MPa above the 292 MPa stress limit, the proportional back-off
(`backoff_gain * (violation - 1)` ≈ 0.002–0.006) reduced the effective
evolution rate to ~3e-5–9e-5, and the volume target said *grow*. Yet the
alive volume fell by ~0.0015 of V0 every iteration, driving sigma_max slowly
further up (292.6 → 293.7 MPa). Three interlocking, now-verified mechanisms:

1. **The leak.** `LevelSet.update` bisects its threshold shift `tau` so the
   thresholded phi field keeps **exactly** the target volume — and then the
   mask is pruned by removal-only post-passes: `keep_connected` inside
   `update`, and the loop's `_morph_open` (`min_member_layers: 1`) +
   `keep_connected` ([loop.py:890-899](../oropt/loop.py)). The volume
   controller never sees what those passes removed, and `self.phi` is
   **never re-synced** to the pruned mask
   ([levelset.py:218-221](../oropt/levelset.py)): at the next iteration the
   bisection budget is charged for *phantom* volume (phi-alive, mask-dead)
   and `tau` erodes real interface material to pay for it — where
   `_morph_open` shaves a fresh one-element fringe again. The one-time prune
   becomes a permanent ~150-volume-units/iteration erosion, ~50× the
   back-off's requested recovery (vf × 3e-5 ≈ 3 units), so the controller
   can never win.
2. **The squash.** The nodal velocity is normalised by its global max, which
   sits in the stress-excluded, protected load-introduction region (measured
   on the run's own checkpointed sensitivity: the argmax element *is*
   stress-excluded; 73 % of alive elements are below 1 % of the max; median
   normalised velocity 0.0014). Meanwhile `tau` equilibrates against the
   velocity at the only active interfaces (the box fringes): the replayed
   iteration measures tau ≈ 0.0996. Net effect: **the entire low-energy
   interior of the part sinks by ~0.1 per update** inside the ±1 plateau, on
   pure controller arithmetic, with no mechanics behind it.
3. **The collapse.** Starting from the +1 plateau, ~0.098/update × 8 updates
   ≈ the full plateau height — and exactly at the 8th update (after iteration
   7, recovered from the salvaged checkpoint) the threshold swept through the
   lowest-energy regions wholesale: 38,175 elements removed in one step, 99 %
   of them in the bottom energy quartile, in a few large connected chunks —
   thin webs 0.8–1.4 mm thick, 35–60 mm away from the growth boxes. The
   volume was the *same* ~0.0015 · V0 leak budget (the collapsed elements are
   fine-mesh tets ~11× smaller than average) redirected into structurally
   arbitrary interior holes. The run was stopped before solving that design;
   a solve would most likely have diverged or spiked sigma.

The level-set has **no useful hole-nucleation mechanism** (H1, confirmed
below), so none of this removal ever happened where a stress-limited design
actually wanted it — and the constraint stayed pinned while the optimiser
disassembled the growth-box fringes and, eventually, the part's own webs.

## The observed trajectory, against what the controller asked for

From `history.csv` (V0 = 104,332; `gate_target_vf` with the config above;
violation = sigma_max/292, the worst ratio — displacement never governed):

| iter | vf actual | sigma_max | feasible | er_eff for next step | target vf next | vf actual next | leak (of V0) |
|-----:|----------:|----------:|:---------|---------------------:|---------------:|---------------:|-------------:|
| 0 | 0.958996 | 278.815 | yes | 0.006773 (damped)    | 0.952501 | 0.950538 | −0.001963 |
| 1 | 0.950538 | 278.936 | yes | 0.006711 (damped)    | 0.944159 | 0.942229 | −0.001930 |
| 2 | 0.942229 | 292.621 | no  | 0.0000319 (back-off) | 0.942259 | 0.940816 | −0.001443 |
| 3 | 0.940816 | 292.434 | no  | 0.0000223 (back-off) | 0.940837 | 0.939296 | −0.001541 |
| 4 | 0.939296 | 292.558 | no  | 0.0000287 (back-off) | 0.939323 | 0.937732 | −0.001591 |
| 5 | 0.937732 | 293.012 | no  | 0.0000520 (back-off) | 0.937781 | 0.936244 | −0.001537 |
| 6 | 0.936244 | 293.423 | no  | 0.0000731 (back-off) | 0.936312 | 0.934806 | −0.001506 |
| 7 | 0.934806 | 293.666 | no  | 0.0000856 (back-off) | 0.934886 | 0.933355 | −0.001531 |

(The iteration-8 vf is measured from the salvaged checkpoint; that design was
never solved — the run was stopped there.)

Two things stand out:

* From iteration 2 on, **the target is above the current vf** (the gate asks
  to add material back) and the actual vf still falls by 0.0014–0.0016 of V0
  (≈150 volume units ≈ 1.3–1.5k elements — the removed elements average ~0.11
  volume units, ~2.4× the design-space mean, i.e. the coarser expansion-mesh
  tets at the box fringes).
* The same leak already existed while feasible: iterations 0→1 and 1→2
  overshot their damped targets by ~0.0019. One mechanism explains both.

Removal location (measured by diffing the archived iteration decks): of
29,769 elements removed over iterations 0–6, 29,105 lay inside a growth
region, 628 within 1 mm of one, 36 beyond (max 1.6 mm ≈ 2 element layers).
Only 659 expansion elements were ever grown. The optimiser spent seven
~2.5-hour iterations exclusively rearranging the growth-box fringes.

## Verified mechanism chain

### H3 — the leak (verified; this is the root cause)

**Where the volume goes.** `LevelSet.update`
([levelset.py:192-221](../oropt/levelset.py)) evolves phi, then `_solve_tau`
bisects the uniform threshold shift so the kept removable volume is the
largest value ≤ `target_vf*V0 - protected_V` — the volume controller is
*exact* (64 bisection steps; measured floor gap ~2e-5 of V0). But the mask
that actually survives the iteration is produced by three removal-only passes
that run *after* the controller:

1. `mesh.keep_connected(alive, anchor)` inside `update`
   ([levelset.py:220](../oropt/levelset.py)),
2. `_morph_open` via `apply_manufacturing` (`min_member_layers: 1`,
   [loop.py:896-898](../oropt/loop.py), [manufacturing.py:147-165](../oropt/manufacturing.py)),
3. `mesh.keep_connected` again ([loop.py:899](../oropt/loop.py)).

On the synthetic reproduction ([levelset_stuck_repro.py](levelset_stuck_repro.py),
a bar + void growth slab driven through the identical code chain with the
run's knobs), the decomposition over the pinned-infeasible iterations is
unambiguous:

```
full chain     : mean leak/iter = -0.01086 of V0 (open 0.01085, keep_connected 0.00000, tau floor 0.00002)
no morph_open  : mean leak/iter = -0.00002 of V0 (the _solve_tau floor gap only)
phi re-synced  : mean leak/iter = -0.00689 of V0, decaying -0.01057 -> -0.00088
```

`_morph_open` **is** the leak. The level-set iso-surface, thresholded on the
mean nodal phi over a tet mesh, always carries a jagged one-element-thick
skin; erosion (any alive element with a dead node-neighbour dies) strips it,
dilation only restores elements that had a surviving neighbour, so every
iteration the open shaves the freshly exposed staircase fringe. The removals
in the synthetic case sit 0–4 element layers from the void interface (3,033
at 0–2 layers, 1,220 at 2–4, none deeper) — the same signature as the real
run's 29.1k/628/36 split.

**Why it never stops.** `update` stores `self.phi` *before* the mask is
pruned and never learns what the post-passes removed. The pruned elements
keep mean-phi ≥ 0, so `_removable_vol_at` counts them as kept: the next
bisection's budget (computed by the loop from the *actual*, pruned vf) is
smaller than the phi-implied volume by exactly the pruned amount. The
instrumented repro shows `phantom(it) == dOpen(it-1)` every iteration: tau
must shift positive and erode that much *real* interface material, which the
open then re-shaves. With phi re-synced each iteration (the `resync_phi`
variant) the same chain anneals — the leak decays 12× within a dozen
iterations — while with the desync it stays constant forever. **The desync is
what converts a one-time morphological cleanup into a permanent erosion
ratchet.**

**Real-scale confirmation.** Replaying one full update+prune iteration from
the salvaged checkpoint on the run-era 2,272,868-element mesh
([levelset_stuck_replay_real.py](levelset_stuck_replay_real.py)):

```
tau = 0.09963  bisection floor gap (budget - achieved) = 3.80e-07
thresholded phi mask:            0.933435
after update keep_connected:     0.933435 (drop 0.000000)
after _morph_open:               0.932176 (drop 0.001259)
after final keep_connected:      0.932176 (drop 0.000000)
target_vf 0.933435 -> actual 0.932176 (leak -0.001260)
```

`_morph_open`'s single-iteration bite (0.00126 of V0) reproduces the run's
observed 0.0014–0.0016 leak; both connectivity passes drop nothing; the
bisection floor is seven orders of magnitude below the leak.

**Why the controller cannot recover.** With `backoff_gain 1.0` and a 0.2–0.6 %
stress violation, the infeasible "grow" step is `ER * (v-1)` ≈ 3e-5–9e-5 of
vf — two orders of magnitude below the leak. The gate's arithmetic is
correct; it is simply outgunned. (With the default `backoff_gain 0.0` the
binary gate would have stepped +ER = +1.5 % per infeasible iteration and
masked the leak by brute force.)

**The endgame — the run's own last update (a_7 → a_8, from the archives).**
Diffing the archived iteration-7 deck against the checkpoint mask:

```
a_7 alive 2036666, a_8 alive 1998638, net -38028, grown 147
removed 38175 (volume 0.00150 of V0, mean elem vol 0.0041 vs design mean 0.0459;
               301 inside a growth region, 29 within 1 mm, 37845 beyond, max 60.2 mm)
removed & protected: 0;  components (largest 5): [19783, 10820, 5800, 605, 321]
sens percentile of removed: median 9.5;  99% in the bottom quartile of energy
sharing a node with a surviving element: 48.5% overall, 39-100% per large component
```

This is the plateau collapse, not more fringe shaving and **not** island
amputation: a `keep_connected` island shares no node with the surviving alive
set, yet every large removed component touches survivors — they crossed the
threshold. The removed volume is the same steady ~0.0015 · V0 leak budget,
but the collapsing plateau handed the bisection tens of thousands of *tiny*
(fine-mesh, ~0.3 mm) elements just above the threshold, so the same volume
bought 25× the element count — entire thin webs (extents like
14.8 × 1.4 × 7.9 mm, i.e. ~1.4 mm thick walls) 35–60 mm from any growth box.
The collapse timing is arithmetically consistent: the plateau starts at +1
and sinks at tau − Vn_local ≈ 0.0996 − 0.001 ≈ 0.098 per update (with the
two larger feasible-removal steps early on), reaching zero at — the 8th
update.

### H1 — no hole nucleation (verified on the run-era code, with two corrections)

The hypothesis said phi sits *at the ±band_width clamp* away from interfaces
and the clamp keeps it there. Measured (on the run-era `_init_phi`; PR #57
has since replaced it), the detail is different but the conclusion stands:

* `_init_phi` scatters a ±1 indicator and smooths it — values live in
  [−1, +1], so the ±3 clamp is a **no-op at init** (0 % of nodes at the
  clamp; 80 % of bulk elements at mean-phi = +1.0 exactly in the repro). The
  plateau is binary at ±1, not ±band_width.
* Away from an interface the only downward forces on phi are the uniform
  −tau shift (which equilibrates against `dt*Vn` at the *interface*) and the
  Laplacian smoothing diffusing the interface deficit inward at
  ~√(passes·iters) element layers. In the repro (no-mfg case, 50 iterations)
  the first element 2–4 layers below the interface crosses phi=0 at
  iteration 15, the 4–6-layer band never crosses, the 6–8-layer band ends at
  +0.39. The real run's *fringe* removals reached at most 1.6 mm ≈ 2 element
  layers past the boxes in 7 iterations.
* The part's free surface has no void elements beyond it in the design
  space, so there is no interface there for tau to move and nothing for the
  threshold to trade against — surface sculpting of the part is impossible
  by construction.
* What the level-set *does* have, as the real run demonstrated, is
  **indiscriminate bulk hole formation by plateau drift**: since tau ≈ the
  interface-local velocity (~0.1) while the squashed bulk velocity is ~0.001
  (H2), every low-energy region sinks ~0.1/update and the whole bottom of
  the energy distribution crosses the threshold *simultaneously* after
  ~1/tau ≈ 8–10 updates. That is not nucleation an optimiser can use — it is
  volume-bisection shrapnel, uncorrelated with the constraint state, landing
  in thin webs the stress field happened to load lightly.

This confirms the earlier measurement (29.1k of 29.8k removals inside the
boxes) and the guidance that stood at the time: **the run-era level-set could
only usefully remodel existing void interfaces — and its plateau drift was
not even neutral, it was destructive on a ~10-iteration timescale.** PR #57
has since replaced the binary init with an energy-rank spread plus a
nucleation term, which turns the indiscriminate drift into deliberate,
energy-ordered interior removal (see "Status relative to PR #57").

### H2 — sensitivity normalisation squash (verified; it also times the collapse)

`update` normalises the nodal velocity by its global max
([levelset.py:204-207](../oropt/levelset.py)). Measured on the run's own
checkpointed sensitivity field (filtered + history-blended) over the alive
elements:

```
max 0.264993 | p99.9 0.0664596 | p99 0.0321675 | p95 0.0127978 | median 0.000635639
max / p99 = 8.2x ; max / median = 416.9x
argmax element: centroid [12.76 -5.11 -52.74], stress-excluded? True, protected? True
alive elements with sens < 1% of max: 72.9% ; < 5% of max: 95.2%
normalised nodal velocity Vn: p50 0.00137, p99 0.1291, max 1.0
```

The stress-exclusion feature deliberately leaves the sensitivity untouched
([loop.py:72-87](../oropt/loop.py)), so the load-introduction energy peak —
whose *stress* is excluded from feasibility precisely because it is a
modelling artefact — still owns the normalisation. The verified consequences:

* The bulk of the part moves at `dt*Vn` ≈ 0.0014/iteration inside a ±1
  plateau; the phi evolution is dominated by tau and smoothing, not by the
  mechanics, so seven ~2.5 h iterations barely changed sigma_max.
* The squash sets the *collapse clock* (see H1/H3): plateau sink rate =
  tau − Vn_local ≈ 0.0996 − 0.001. A flatter normalisation (or a velocity
  with zero mean over the alive set) would not sink the interior at all.

The original hypothesis framed this as "the bulk gets Vn ≈ 0" — verified,
with the addition that the squash is not merely a slow-down: combined with
the tau equilibrium it *schedules* the interior plateau collapse.

### Falsified / immaterial candidates

| Candidate | Verdict |
|---|---|
| `_solve_tau` floor semantics ("largest volume ≤ budget") | Real but immaterial: gap ~2e-5 of V0 in the synthetic repro and 3.8e-7 of V0 on the real mesh — up to 7 orders below the leak. |
| `keep_connected` island drops as the leak | Falsified: exactly 0.000000 of V0 in both the synthetic steady state and the real-scale replayed iteration; the 38k-element final removal is also not island-dropping (every large component touches survivors, and 0 protected elements were removed). |
| phi drift near the box interface outpacing the tau shift | Falsified as a *leak*: any drift is inside the bisection's volume accounting; with the post-passes disabled the mask tracks the target to 2e-5 of V0 for 50 iterations. (Global plateau drift is real and causes the collapse — but it is volume-neutral, not the leak.) |
| clamp degeneracy at ±band_width | Falsified as stated (plateau is at ±1; the clamp is a no-op at init) — superseded by the plateau-drift/diffusion-creep picture above. |

## Side findings

* **phi is not checkpointed.** `save_checkpoint` stores only
  (iteration, alive_mask, sens_prev) ([status.py:204-208](../oropt/status.py)).
  Any resume re-initialises phi from the alive mask via `_init_phi`, whose
  smoothing disagrees with the mask it was built from: on the real mesh the
  implied mask differs by 5,088 elements (0.00099 of V0) before the update
  even runs. A stopped/resumed level-set run silently perturbs the design —
  and it also resets the sunken plateau back to +1, restarting the collapse
  clock, which is why the replayed iteration shows fringe behaviour rather
  than a second collapse.
* **The run-era starter deck is gone.** The follow-up BESO run's PREPARE
  regenerated the root starter decks in place at 09:09 (2,285,778 design
  elements vs the run's 2,272,868 — same part, new expansion mesh), so this
  analysis reconstructs the run-era mesh from
  `implicit_elevator-linkage_pull_0000.rad.pre-brick-fix.bak`. Salvaged run
  artifacts (checkpoint, history, configs, all iteration archives and VTUs)
  are frozen in `opti_run1\_levelset_salvage_20260706\`.

## Status relative to PR #57, and remaining fix requirements

The fix chip that existed when this analysis started ("Level-set hole
nucleation + backoff floor") was implemented as PR #57 while the analysis was
in progress. Mapping this analysis onto it:

**Landed in #57 (confirmed by this analysis as necessary, but not sufficient):**

* **Back-off floor** — `er_eff = ER * max(backoff_floor, min(gain*(v-1),
  cap))`, default floor 0.25. Exactly the requirement derived here: at a
  0.2 % violation the run-era step (3e-5) could not outpace even benign
  noise, let alone the leak. Caveat: whether the floor out-runs the leak is
  *geometry-dependent*. Replaying the checkpoint iteration on the real mesh
  under post-#57 defaults nets **+0.00206 of V0** (target +0.00375, prune
  leak −0.00144) versus the run-era **−0.00118** — the stall mode is masked
  on this deck. On the synthetic repro (proportionally larger fringe) the
  leak still wins and vf still ratchets down under pinned infeasibility. The
  floor masks the stall; it does not remove the defect.
* **Hole nucleation** — energy-rank `_init_phi` spread over (0, ±band_width]
  plus the `nucleation_rate` reaction term `Vn - rate*(1-Vn)`. This replaces
  the run-era binary ±1 plateau, so the "indiscriminate plateau collapse"
  hazard (H1/H3 endgame above) is transformed: the interior *is* now meant to
  sink, but in filtered-energy order from iteration 0 instead of all-at-once
  after 8 ticks of a drift clock. Measured on the real mesh (same replayed
  iteration, post-#57 defaults): the update removes 96,941 interior elements,
  100 % of them in the bottom energy quartile and none inside a growth region
  — targeted low-energy carving, exactly the intent. Operational caveat: the
  removal arrives as two huge contiguous sheets (56,662 and 35,604 elements,
  ~25 × 2–6 × 10–16 mm), so the first solve after such a step carries a real
  severance/divergence risk — see the stall/step-size guard below. The H2
  squash also still compresses the *velocity* ordering during evolution (the
  rank init carries most of the discrimination), and the normalisation peak
  still sits in the stress-excluded protected load region.

**Still open on `master` at the time of this analysis — since closed by
`fix/levelset-volume-leak` (the primary defect of this run):**

1. **The volume controller must see what the iteration actually keeps.**
   Measured post-#57: `_morph_open` still leaks 0.00144 of V0 on the real
   mesh (0.0103 on the synthetic case) per iteration against a grow target,
   and the phi-desync ratchet (`phantom(it) == dOpen(it-1)`) is unchanged.
   Two complementary requirements:
   * Re-sync phi to the final post-prune mask (or equivalently charge
     `_solve_tau`'s budget only for volume the mask actually has). Without
     this, any removal-only post-pass (manufacturing, connectivity) turns
     into a permanent per-iteration erosion — ~0.0015 of V0 per iteration on
     the run.
   * Account the prune loss in the next target (e.g. add the measured
     post-pass removal back onto `target_vf`), so `_morph_open` becomes
     volume-neutral over the iteration pair instead of compounding. The
     `resync_phi` experiment shows re-syncing alone shrinks the leak 12× and
     lets the fringe anneal; both together should close it.
2. **Checkpoint phi** alongside the alive mask so resume neither perturbs
   the design (5,088-element flip measured on the real mesh) nor resets the
   field. Post-#57 this matters *more*: a resume re-initialises via the
   energy-rank spread, which re-orders the entire field by the current
   sensitivity — a much larger silent perturbation than the run-era smoothing
   flip.
3. **Stall detector in the loop.** `target_vf > vf` (a grow request) combined
   with `vf` falling for N consecutive iterations is controller-defeating
   behaviour no run should silently continue through; log it loudly and/or
   stop with a diagnostic after ~3 occurrences. A companion guard — "one
   update removed > K× the recent per-iteration element count" — would have
   caught the web collapse before a 2.5 h solve was spent on it. This also
   covers the geometry-dependent case where the #57 floor loses to the leak.

## Reproduction

Both scripts demonstrate the defect and are era-pinned: run them at commit
`2280f86` (the analysis-era master; on later code the leak they assert is
fixed and their instrumented replica of `update()` diverges from the real
one). The run-era (pre-#57) semantics
are reproduced bit-identically by seeding the old binary-indicator phi
through the public `opt.phi` (exactly how a resume seeds state) and zeroing
`nucleation_rate` / `backoff_floor`; each script additionally runs a
post-#57-defaults leg to show which behaviours changed and that the prune
leak did not.

* [docs/levelset_stuck_repro.py](levelset_stuck_repro.py) — synthetic
  bar+growth-slab case through the exact production code chain
  (`gate_target_vf → LevelSet.update → apply_manufacturing →
  keep_connected`), with the update internals instrumented and asserted
  bit-identical to `update()`. Prints the leak decomposition, the
  removal-location bands, the H1 plateau/creep table, and asserts the
  mechanism (leak < 0 under grow targets, leak == post-pass removals, leak
  vanishes without `_morph_open`, phantom == previous open, no deep
  nucleation in 50 iterations, leak still present under post-#57 defaults).
  Runs in ~1 min, no run artifacts needed.
* [docs/levelset_stuck_replay_real.py](levelset_stuck_replay_real.py) — the
  real-scale evidence: reconstructs the run-era design space from the
  salvaged artifacts, diffs the run's own last update (a_7 → a_8), measures
  the H2 squash on the run's checkpointed sensitivity, and replays one full
  update+prune iteration on the 2.27M-element mesh with the same
  instrumentation, under both run-era and post-#57 semantics. Needs the
  salvage folder (see file docstring).

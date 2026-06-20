# Research spike: density-based SIMP + Optimality-Criteria for `oropt`

*Status: investigation + offline prototype + recommendation. Nothing here is wired
into the live OpenRadioss loop.*

---

## TL;DR — recommendation: **NO-GO** (stay discrete: BESO / TOBS / level-set)

| Question | Finding |
|---|---|
| Is the SIMP **sensitivity** obtainable from OpenRadioss? | **Yes.** `dC/dρ_e = -(E'/E)·2·U_e` and `U_e` is exactly `Results.energy` (`/ANIM/ELEM/ENER`). Proven against finite differences in the prototype. |
| Can OpenRadioss apply a per-element **E(ρ)=E0·ρ^p** modulus cheaply? | **No.** There is no element-wise modulus field. The only route is *quantising* ρ into a discrete set of `/MAT` cards and **re-assigning every element to a per-level `/PART`** each iteration — a large escalation over today's verbatim "omit deleted cards" rewrite, and it breaks the clean `/SURF/PART/EXT` contact regeneration. |
| Will soft (low-ρ) elements survive the implicit **MUMPS** solve? | **Risky.** They keep the tangent non-singular (a plus — no free-node pinning), but they inflate conditioning and, more fundamentally, the model is **nonlinear elasto-plastic (LAW36) with contact**, where SIMP's linear-elastic compliance theory — including eq. (1) — does not hold. |
| Can the grayscale be driven to near **0/1** (manufacturable)? | **Yes**, with penalisation `p≈3` + Heaviside projection + β-continuation. The prototype goes from grayness 0.95 → 0.02. But the final clean field still needs a *threshold to a discrete mesh* — i.e. you end back at element deletion. |

**Why no-go.** The single hard prerequisite people assume is missing (sensitivities)
is actually *available*. Everything else, though, fights SIMP on this specific
model: OpenRadioss has no per-element modulus, so the deck rewrite becomes
invasive and contact-regeneration breaks; the solve is nonlinear+plastic+contact,
so the linear-elastic compliance objective and its tidy sensitivity are physically
mismatched; and the project already ships **three** discrete optimisers (BESO,
level-set, TOBS) that reuse the verbatim element-deletion deck path and give
manufacturable black-and-white designs for far less risk. The optimiser *maths* is
cheap and de-risked (this prototype), but the cost and risk live entirely on the
OpenRadioss / deck / physics side, and they are not justified.

**When to revisit.** Two things would flip the call: (1) a **linear-elastic** proxy
model (drop LAW36 plasticity + contact for the optimisation pass), and (2)
OpenRadioss exposing a genuine per-element modulus (or accepting many parts without
wrecking contact). Until both hold, discrete is the right tool here.

---

## 1. Context: where SIMP *would* slot in

Since this spike was scoped, `oropt` grew a real **optimiser seam** (PRs #9–#14).
`loop.build_optimizer(cfg, mesh, protected, anchor)` now selects one of three
optimisers by `cfg.optimizer` (`"beso"` / `"levelset"` / `"tobs"`), and they all
implement the **same binary contract**:

```python
class Opt:
    def __init__(self, mesh, cfg, protected_mask, anchor=None): ...
    def volume_fraction(self, alive_mask) -> float
    def raw_sensitivity(self, results, elem_ids, alive_mask) -> np.ndarray   # map_sensitivity
    def filter_history(self, raw, sens_prev) -> np.ndarray                   # blend_history
    def next_target_vf(self, current_vf, feasible) -> float
    def update(self, alive_mask, sens, target_vf) -> alive_mask              # boolean -> boolean
```

The key word is **binary**: the state passed in and out of `update()` is a boolean
`alive_mask`, and after the update the loop writes the deck by *omitting deleted
element cards only* (`deck.Deck.write`). BESO thresholds by sensitivity, TOBS picks
flips with an ILP, level-set thresholds a nodal field — but all three reduce to a
0/1 mask, reuse `Mesh.keep_connected`, the free-node pinning, and the contact skin
that OpenRadioss regenerates from the survivors. Adding a fourth *discrete*
optimiser is cheap precisely because the seam already carries everything it needs.

**SIMP does not fit this seam.** Its design variable is a *continuous* density
`ρ_e ∈ [0,1]^N` that the solver must physically see as a modulus `E_e = E0·ρ_e^p`.
That means:

* a **wider seam** — `ρ` (not a bool mask) threaded through `loop` → `status` →
  checkpoint → deck, plus continuation state (β, p);
* a **new deck path** — emit a per-element modulus, not omit cards;
* the AM constraints in `manufacturing.py` (which operate on the alive mask) and
  the multi-load weighting (`combine_sensitivity`) would both need density-based
  reformulations.

So "add SIMP like we added TOBS" understates the work by a wide margin. The rest of
this document quantifies the deck and physics side.

## 2. The sensitivity *is* available (the one piece of good news)

For a structure with `K(ρ) = Σ_e K_e(ρ_e)` and each element matrix scaling linearly
with its modulus, `K_e(ρ_e) = (E(ρ_e)/E_ref)·K_e^0`, the exact compliance gradient is

```
dC/dρ_e = -Uᵀ (dK_e/dρ_e) U
        = -(E'(ρ_e)/E(ρ_e)) · (Uᵀ K_e U)
        = -(E'(ρ_e)/E(ρ_e)) · 2 · U_e            (1)
```

where `U_e` is the **element strain energy** — exactly what OpenRadioss writes to
`/ANIM/ELEM/ENER` and `results.Results.energy` already parses. No adjoint, no
surrogate, no finite differencing. (For several load cases it is the weighted sum
of the per-case energies, since (1) is linear in `U_e` — the same idea as
`beso.combine_sensitivity`.)

`oropt/simp.py::compliance_sensitivity` implements (1) and the prototype's
`test_compliance_sensitivity_matches_finite_difference` confirms it reproduces the
central-difference gradient of the synthetic compliance to `< 1e-4` relative error.

> **Subtlety worth flagging.** `U_e` in (1) is the *penalised, measured* strain
> energy. The textbook form `dC/dρ = -p·ρ^(p-1)·2·u` is identical but uses the
> *unpenalised* reference energy `u = U_e/ρ^p` (and `Emin=0, E0=1`). A black-box
> solver reports the measured energy, so the measured-energy form (1) is the one to
> wire up. Getting this wrong silently rescales the gradient by `ρ^p`.

**The catch (see §4):** (1) is a *linear-elastic* identity. The real model is LAW36
plasticity + contact, where the reported energy includes plastic dissipation and
contact work, and "compliance" is not even the objective (the loop minimises mass
under σ/displacement limits). So the clean sensitivity is only an approximation of
the actual physics — and a worse one as elements yield.

## 3. Can OpenRadioss apply `E_e = E0·ρ_e^p` cheaply?

Short answer: **no element-wise modulus field exists**, so all three options below
reduce to "fake a continuous field with discrete materials," at escalating deck
cost. Recall how the deck is structured (what `deck.py` already parses): each
**element** lives in a `/TETRA4/<part_id>` block; each **`/PART`** binds exactly one
`/PROP` and one `/MAT`; **`/MAT/LAW36/<id>`** carries the Young's modulus as a scalar
field. Modulus is therefore a *material* property, reachable only *via the part an
element belongs to*.

### (a) Discrete `/MAT` quantisation + per-element `/PART` reassignment — the only real route

Quantise `ρ` into `Q` levels `ρ_1…ρ_Q`; emit `Q` materials with
`E_k = Emin + ρ_k^p·(E0−Emin)`, `Q` parts (each `prop + mat_k`), and partition the
design elements into `Q` `/TETRA4/<part_k>` blocks by quantised density. What
`deck.py` would have to emit, *beyond today's "copy verbatim minus deleted cards"*:

* `Q` new `/MAT/LAW36/...` cards (cloning the design material, overriding `E`, and —
  because LAW36 is elasto-plastic — deciding how/whether to scale the yield curve,
  which SIMP gives no guidance on);
* `Q` new `/PART/...` (and possibly `/PROP/...`) cards;
* a **re-partition of every design element** into its level's block *each iteration*
  (vs. today's single-block filter). This is the same O(N) line-emit cost, but it
  rewrites *all* design element cards rather than dropping a few.

**This breaks contact regeneration.** The linkage contact master is
`/SURF/PART/EXT` on the *single* design part id. Split the design into `Q` parts and
that surface no longer spans the design domain; you would have to emit a
`/SURF/PART/EXT` (or `/SURF/SEG`) enumerating all `Q` parts and keep it in sync as
elements migrate between levels every iteration. The current architecture's
headline simplification ("contacts need no edits") is lost.

**Re-conversion cost.** Strictly, no external re-run of `k_to_rad_converter` is
needed — these are additive edits to the already-converted `_0000.rad`. But the
bookkeeping (materials, parts, properties, the contact surface, and keeping element
ids ↔ parts consistent) is a different order of complexity from the current editor,
and `Q` is a real knob: small `Q` quantises the modulus coarsely (hurting the OC
update's accuracy), large `Q` multiplies parts/materials and bloats contact.

### (b) `/PROP`-level scaling — not available

OpenRadioss solid properties (`/PROP/SOLID`, TYPE14, …) parameterise element
*formulation* (integration, hourglass, `Ismstr`, …), **not** a modulus multiplier.
There is no per-property `E` scale factor to exploit, so (b) collapses into (a)
(you would still need a distinct material per level).

### (c) A genuine element-wise modulus field — not supported

For standard Lagrangian solids OpenRadioss exposes no "per-element E" or
field-driven modulus (the kind some research codes provide). Initial-state /
reference-geometry mechanisms don't scale stiffness. So (c) does not exist today; if
it ever did, SIMP's deck story would become as clean as element deletion and this
recommendation would change.

**Net:** the deck side goes from *trivial* (today) to *invasive with a broken
contact invariant*. That is the dominant cost of SIMP here.

## 4. Conditioning, soft elements, and contact

* **Tangent stays non-singular (a genuine plus).** Because SIMP *keeps* every
  element at `ρ ≥ ρ_min` (modulus `Emin > 0`), there are no element-less free nodes,
  so the free-node `/GRNOD`+`/BCS` pinning that element deletion needs is
  unnecessary. MUMPS (a sparse *direct* solver) also tolerates the resulting
  ill-conditioning far better than an iterative solver would.
* **…but soft elements still hurt.** `Emin` must stay well above zero
  (`~E0·1e-6…1e-9`) or tiny pivots and a ballooning condition number degrade the
  Newton convergence of the *nonlinear* implicit solve. Near-void elements can also
  distort/hourglass under load, polluting the very energy field that drives (1).
* **The physics mismatch is the real problem.** SIMP compliance theory is
  linear-elastic. This model is **LAW36 plasticity + contact**, solved fully
  nonlinear every iteration (the README is explicit: "no linear-elastic
  simplification"). Once elements yield, `U_e` is elastic + plastic energy, eq. (1)
  is no longer the exact gradient, and "compliance" isn't the objective anyway
  (mass-min under σ/d limits is). SIMP would be optimising a linear surrogate of a
  nonlinear response.
* **Contact on soft material is meaningless.** With deletion, the contact skin
  regenerates around real, load-bearing survivors. With SIMP, *nothing is removed*,
  so `/SURF/PART/EXT` always wraps the **original full boundary** — the optimised
  shape never becomes the contact surface, and soft elements at the skin carry
  contact pressure they physically shouldn't. For a model whose load is *applied
  through contact* (6 kN via rigid cylinders), this is a first-order correctness
  problem, not a detail.

## 5. Driving grayscale to 0/1 (manufacturability)

This part works, and the prototype demonstrates it. With penalisation `p=3`, a
density filter (radius ≈ 1.5× element size), and a **Heaviside projection** with
β-continuation, intermediate densities are pushed to the rails:

```
$ python -m oropt.simp
iterations            : 120
compliance  start->end: 0.1396 -> 0.0119      # ~12x stiffer at fixed volume
volume fraction (final): 0.400  (target 0.400)
grayness    start->end: 0.953 -> 0.021         # ~pure black/white
final beta            : 8.0
```

`grayness = mean(4ρ(1−ρ))` (0 = pure 0/1, 1 = all-0.5) collapses to 0.02 — a
manufacturable field. **But** note the implication: a near-0/1 result then needs a
**threshold to a clean discrete mesh** for CAD/print — at which point you have
re-derived element deletion, just with a more expensive path to get there. The
manufacturability argument for SIMP over BESO/TOBS is therefore weak *for this
project*, where the deliverable is already a discrete optimised mesh (smoothed
surface export, `smooth.*`).

## 6. Effort / risk summary

| Area | BESO / TOBS / level-set (today) | SIMP (this spike) |
|---|---|---|
| Sensitivity source | `Results.energy` (BESO number) | `Results.energy` via eq. (1) — **same, available** |
| Optimiser maths | shipped | prototype done here, **low risk** |
| Loop/seam state | boolean `alive_mask` | continuous `ρ` + continuation — **seam must widen** |
| Deck rewrite | omit deleted cards (trivial) | `Q` mats/parts + per-element re-partition — **invasive** |
| Contact skin | regenerates from survivors (free) | `/SURF/PART/EXT` breaks across `Q` parts — **must rebuild** |
| Implicit conditioning | excellent (deletion) | soft elements OK-ish with `Emin`; **nonlinear+plastic mismatch** |
| Objective fidelity | uses true nonlinear σ/d limits | linear-elastic compliance surrogate — **physically off** |
| Manufacturability | discrete by construction | needs `p`+Heaviside, then a threshold back to discrete |

The only **low-risk** column is the optimiser maths — which is exactly the half this
prototype already de-risks.

## 7. Recommendation (detail)

**Do not pursue SIMP for the production elevator-linkage model.** Stay with the
discrete optimisers. They already reuse the verbatim-omit deck path, keep the
contact invariant, run against the *true* nonlinear σ/displacement constraints, and
produce manufacturable black-and-white designs. TOBS in particular gives much of
SIMP's "principled volume-constrained step" via its ILP without leaving the binary
world.

If SIMP is ever revisited, the cheapest informative experiment — *before* touching
`deck.py` — is a **conditioning/contact probe**: take one converted deck, split the
design part into a handful (`Q≈4`) of quantised-modulus parts with a soft `Emin`
level present, rebuild `/SURF/PART/EXT` over all parts, and check that (a) MUMPS
still reaches NORMAL TERMINATION and (b) contact behaves sanely with soft skin
elements. That single solve (~13 min) answers the two questions that actually gate
SIMP here; the optimiser side (this prototype) is already known to work.

## 8. The prototype and its tests

* `oropt/simp.py` — **EXPERIMENTAL**, not imported by the loop. Implements the SIMP
  interpolation `E(ρ)=Emin+ρ^p(E0−Emin)`, the energy→sensitivity map (1), a density
  filter mirroring `mesh.filter_matrix`, the Heaviside projection (+ derivative),
  the OC bisection update, a synthetic (parallel-spring) compliance model standing
  in for OpenRadioss, and a driver. Run `python -m oropt.simp` for the demo above.
* `tests/test_simp.py` — hermetic, analytic. Verifies: the sensitivity matches
  finite differences; the OC bisection meets the volume target with densities in
  `[0,1]`; one OC step reduces the synthetic objective under the volume constraint;
  the density filter is a volume-preserving partition of unity; and the Heaviside
  projection drives grayscale toward 0/1.

These prove the optimiser mathematics in isolation. They deliberately say nothing
about OpenRadioss — because, as §§3–4 argue, that is where SIMP would actually break.

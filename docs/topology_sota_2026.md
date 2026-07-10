# Research spike: state of the art in topology optimisation (2020–2026) — what's worth implementing in oropt

*Status: deep-research survey + ranked implementation shortlist. Nothing here is
wired into the loop yet.*

*Method: multi-angle web research (5 search angles → 22 sources fetched → 93
claims extracted → the top 25 adversarially verified by independent 3-vote
refutation panels: 23 confirmed, 2 refuted). Claims below are marked
**[verified]** (survived a 3-vote panel against the primary source) or
**[extracted]** (quoted from a primary source but not put through a panel —
treat as probably-right, re-check before building on the detail).*

---

## TL;DR — ranked shortlist

Everything is judged against oropt's hard constraints: **binary alive-mask
seam** (0/1 per element, deck rewritten by omitting cards), **only sensitivity
= `/ANIM/ELEM/ENER` per-element energy** (plus the per-element von-Mises the
extractor already reads), **no adjoint / no per-element modulus** (SIMP no-go,
see `simp_spike.md`), **~13 min/solve, 50–150 solves per run**, nonlinear
elasto-plastic (LAW36) + contact physics.

| # | Candidate | What it is | Fit with the seam | Energy field enough? | Cost | Verdict |
|---|---|---|---|---|---|---|
| 1 | **SAIP / SCIP** (sequential approximate/conservative integer programming, Liang & Cheng group) | Rigorous 0/1 optimiser: separable integer subproblems solved by an analytic canonical relaxation; SCIP adds MMA-style conservative asymptotes | **Direct** — same shape as TOBS (per-iteration binary subproblem), drop-in fifth optimiser | Yes for the compliance-type baseline (its basic sensitivity *is* `u_e^T k_e u_e`) | Medium (~a `tobs.py`) | **Implement** |
| 2 | **LS-TaSC multipoint method** (Roux) | Controller layer: mass-fraction target and per-load-case weights become *global* variables driven by numerical derivatives built from iteration history | **Direct** — generalises the existing feasibility back-off + fixed multi-load weights; sits above all four optimisers | Yes (local update stays fully-stressed / energy-uniformity) | Medium (loop-level) | **Implement** |
| 3 | **MHCA** (variable-neighbourhood-radius HCA) | Adaptive CA neighbourhood radius as a search schedule | **Trivial** — `hca.py` already builds one fixed-radius filter | Yes | Low (a knob + schedule) | **Implement (cheap)** |
| 4 | **Reaction–diffusion level-set update** (Yamada/Otomori/Choi–style RDE) | Replace HJ-advection-style φ evolution with an implicit reaction–diffusion step; one diffusion knob controls geometric complexity | Good — swaps the interior of `levelset.py`'s evolve step; mask contract unchanged | Yes (reaction term = energy-derived topological-derivative field) | Medium | **Implement (robustness play)** |
| 5 | **MFSE + Kriging/DNN surrogate** (material-field series expansion, non-gradient TO) | Reparameterise topology into ≤50–200 correlated field coefficients; drive with an online surrogate over *scalar* outputs only | Different paradigm — needs a new outer loop, but emits a mask fine; pairs naturally with `fastmode` (3-min linear proxy) | Doesn't even need the field — only σ_max/compliance scalars | High | **Prototype later** |
| 6 | **TDSA sensitivity** (topological-derivative-based discrete sensitivity, stress-quadratic forms) | The *exact* sensitivity for a 1→0 element flip is a linear combination of quadratic forms of the stress components — not raw energy density | Upgrade path for all four optimisers (`sensitivity: tdsa`) | **No** — needs the per-element stress tensor from the anim | Medium + solver-output work | **Research item** |
| 7 | **PTO** (proportional topology optimisation, stress-driven) | Distribute material proportional to a per-element stress power | Direct (fifth optimiser) but overlaps `sensitivity: vonmises` + `addback_stress_bias` | Yes | Low | **Optional** |
| 8 | **ESL / DiESL** (equivalent static loads) | Adjoint-free nonlinear TO via linear static subproblems on deformed geometry | **Blocked** — published TO variant runs SIMP on the linear models → needs per-element modulus, the exact SIMP blocker | n/a | High | **No-go (revisit)** |
| 9 | **Direct-prediction neural-network TO** | Train a net to output designs | n/a | n/a | n/a | **No-go** (DTU review: few real breakthroughs) |

Recommended order: **SAIP/SCIP** and **MHCA** first (one new optimiser + one
cheap HCA upgrade), then the **multipoint controller** and the **RDE level-set
update**, with **MFSE+surrogate** as the exploratory phase-2 bet that attacks
the actual bottleneck (solve count) rather than the update rule.

---

## 1. Discrete/binary methods beyond BESO & TOBS

### 1.1 SAIP — sequential approximate integer programming **[verified]**

The Liang & Cheng group (Dalian UT) has spent 2019–2026 building the rigorous
mathematical-programming counterpart of BESO: topology optimisation posed
*natively* over 0/1 variables and solved as a sequence of **separable
approximate integer subproblems** via a **canonical relaxation algorithm
(CRA)** — an analytic per-element dual update, no branch-and-bound, no LP
relaxation. Scalability to large meshes is a design goal, and there is a
public **128-line MATLAB reference implementation** (SMO 2019) plus a 2025
consolidating review by the originators.

* Review: *Discrete variable topology optimization by sequential approximate
  integer programming*, Engineering Optimization (2025),
  <https://www.tandfonline.com/doi/full/10.1080/0305215X.2024.2445653>
* Foundational papers: CMAME 2019 (method), SMO 2019 (128-line code), SMO 2020
  (trust-region variant for non-monotonic objectives),
  <https://link.springer.com/article/10.1007/s00158-020-02693-2>

**Why it fits.** The per-iteration contract is exactly TOBS-shaped: linearise
around the current design, solve a small binary program with a move limit,
emit a new 0/1 mask. But where TOBS hands the subproblem to a generic ILP
solver (HiGHS) with a linearised volume band, CRA solves its separable
subproblem *analytically* (a dual bisection over one multiplier — microseconds
at 575k elements, no MILP scaling worries) **[verified]**.

**Sensitivity.** For compliance-type objectives the basic SAIP sensitivity
reduces to the element strain energy `u_e^T k_e u_e` — i.e. exactly
`/ANIM/ELEM/ENER` **[verified]**. The refined TDSA sensitivity (below) needs
stress components, but is *not* required to run the baseline method.

### 1.2 SCIP — the conservative variant **[verified]**

*Sequential Conservative Integer Programming* (Sun, Cheng, Zhang & Liang, Acta
Mechanica Sinica 2023/24,
<https://link.springer.com/article/10.1007/s10409-023-23151-x>) replaces the
linear subproblem with a **nonlinear approximate integer subproblem containing
reciprocal variables whose conservativeness is controlled by MMA-style moving
asymptotes**. Two properties matter at 13 min/solve:

* stable optimisation **without relying on an active volume constraint** —
  relevant because oropt's feasibility back-off deliberately moves the volume
  target around, which is exactly when a linearised volume band (TOBS's ε) is
  weakest;
* convergence regulated **without additional structural analyses** — the
  conservativeness update needs no extra FE solves *(2-1 vote: this is the
  authors' numerical observation on linear-elastic benchmarks with analytic
  sensitivities; transfer to a noisy black-box energy field is plausible but
  unvalidated)*.

**Proposed shape in oropt:** a fifth optimiser `saip.py` implementing the same
binary contract as `tobs.py` (`raw_sensitivity` = existing energy mapping;
`update()` = CRA dual bisection with move limit; optional SCIP asymptote state
checkpointed like HCA's `x`). Reuses the filter, history blend, back-off gate,
multi-load combining, manufacturing constraints and `keep_connected`
unchanged. Risk: convergence claims are from linear-elastic analytic-gradient
settings — mitigated by the fact that at worst it degrades to a
TOBS-with-better-subproblem, and the subproblem solver itself is trivially
testable hermetically.

### 1.3 PTO — proportional topology optimisation **[verified, medium confidence]**

A 2025 benchmark (Zheng/Qiu/Chen, *Mathematics* 13(15):2353,
<https://doi.org/10.3390/math13152353>) gives the first systematic
BESO-vs-PTO comparison for stress minimisation, including p-norm aggregation
effects. Results are **mixed** — BESO sometimes reaches lower peak stress at
higher compliance. Two adjacent claims were **refuted** in verification: that
PTO's speed advantage matters at expensive-FE budgets (0-3), and that PTO is
sensitivity-free while BESO is not (1-2; both consume a per-element field).
Since oropt already has `sensitivity: vonmises|blend` and
`addback_stress_bias`, PTO would add little — **optional**, only as a cheap
experiment if stress-dominated runs misbehave.

## 2. The sensitivity itself: TDSA **[verified]**

The same Dalian group derived the **rigorous discrete sensitivity for a 0/1
element flip** (SMO 2022,
<https://link.springer.com/article/10.1007/s00158-022-03321-x>; 3D extension
CMAME 2024,
<https://www.sciencedirect.com/science/article/abs/pii/S0045782524004079>):
elements fall into three classes needing shape-, topological- and
configuration-derivative treatment, and the correct removal sensitivity is a
**linear combination of quadratic forms of the stress components**, not the
raw strain-energy density. Raw `/ANIM/ELEM/ENER` is therefore a
BESO-heuristic *proxy* for the true flip sensitivity — a useful frame for why
BESO needs its filter/history stabilisers.

**Implication for oropt.** If the engine anim can emit the per-element stress
*tensor* (the extractor already reads von-Mises, so tensor output via the
anim keyword family is worth a spike), a `sensitivity: tdsa` mode could feed
*all four* optimisers a more principled ranking. Caveats: the derivation is
linear-elastic; the demonstrated accuracy advantage is mainly for
**non-self-adjoint** objectives (compliant mechanisms) — for compliance-like
runs the energy field is already close. **Research item, not a quick win.**

## 3. Controller-based constrained optimisation: the LS-TaSC multipoint method **[extracted]**

Source: V. Roux, *The LS-TaSC multipoint method for constrained topology
optimization* (LS-DYNA conf.),
<https://lsdyna.ansys.com/wp-content/uploads/2022/12/the-ls-tasctm-multipoint-method-for-constrained-topology-optimization.pdf>.

LS-TaSC (the tool oropt's HCA is modelled on) handles constraints on highly
nonlinear structures **without any per-element derivatives** by splitting the
variables:

* **local** (per-element topology) — updated by the fully-stressed /
  uniform-internal-energy-density rule (what `hca.py` already does);
* **global** (a handful: per-load-case weights + part mass fractions) —
  driven to satisfy the constraints by **numerical derivatives + mathematical
  programming**, where the derivatives come from the *multipoint* scheme
  (response surfaces over points the run has already visited — i.e. iteration
  history, not dedicated perturbation solves).

This is the principled version of two things oropt already half-has: the
feasibility back-off controller (a 1-D proportional rule on the volume
target) and the *fixed* multi-load weights. The multipoint upgrade would (a)
fit a small response surface `constraint(mass target, case weights)` over the
run history, (b) each iteration pick the mass target that is *predicted* to
sit on the constraint boundary instead of reacting to last iteration's
violation, and (c) adapt load-case weights so no case's load path collapses —
addressing the documented ping-pong/glide tuning (`backoff_gain`,
`damping_threshold`) with data the run already produces, at **zero extra FE
solves**.

**Proposed shape:** a loop-level controller module (e.g. `controller.py`)
that wraps `next_target_vf` and `combine_sensitivity`; per-optimiser opt-in
key (e.g. `backoff_mode: multipoint`). All optimisers benefit at once.

## 4. HCA upgrade: MHCA **[verified]**

Afrousheh, Marzbanrad & Göhlich, SMO 2019,
<https://link.springer.com/article/10.1007/s00158-019-02254-2>: the single
algorithmic novelty over standard HCA is a **variable (adaptive)
neighbourhood radius** in the CA update — large early (global search),
shrinking as the design localises — plus the plastic-strain-energy-uniformity
setpoint over the loading history. Benchmarked directly against LS-TaSC and
CrasHCA on crash energy-absorber problems.

**Fit:** `hca.py` builds one fixed-radius neighbourhood
(`self._W = mesh.filter_matrix(cfg.filter_radius)`); MHCA is a radius
*schedule* (rebuild or pre-build 2–3 filter matrices and switch/interpolate),
a config knob (`hca.radius_schedule`) and tests. Caveats: 2019 paper, explicit
crash dynamics vs our implicit statics, and `/ANIM/ELEM/ENER` is total (not
purely plastic) energy — but the mechanism is objective-agnostic.
**Cheapest real improvement available.**

## 5. Level-set advances: reaction–diffusion update **[extracted]**

The Kyoto group's reaction–diffusion-equation (RDE) level-set method
(Otomori/Yamada/Izui/Nishiwaki, canonical MATLAB code paper, SMO 51:1159–1172,
<https://link.springer.com/article/10.1007/s00158-014-1190-z>; FreeFEM
follow-ons through the 2020s) evolves φ with an **implicit reaction–diffusion
step** instead of Hamilton–Jacobi advection:

* no upwinding, no velocity extension, no reinitialisation — the machinery
  whose home-grown equivalents caused oropt's documented level-set
  pathologies (volume leak, velocity-normalisation squash — see
  `levelset_stuck_analysis.md`);
* the reaction term is a topological-derivative-like field (energy-derived
  for compliance), so the existing sensitivity pipeline feeds it;
* **one scalar diffusion parameter explicitly controls geometric complexity**
  (member/hole count) — a knob none of the current four optimisers has, and a
  natural companion to the AM `min_member_layers` constraint;
* natively bi-directional (holes nucleate from the reaction term), unlike
  classic HJ level sets.

**Proposed shape:** a new evolve mode inside `levelset.py`
(`levelset.update_rule: rde`), keeping the bisected volume threshold, the
post-PR-#57/#60 mask-resync and speed-scale fixes, and the checkpointed φ.
The implicit diffusion solve on the nodal graph Laplacian is a sparse CG —
cheap next to a 13-min solve.

Also noted **[extracted]**: Amstutz-style topological-derivative level-set
codes exist in FreeFEM (2D educational, SMO 2023; parallel 3D, FEAD 2025) but
lean on adaptive remeshing, which the fixed-mesh deck path can't use — mine
them for the update formula, not the architecture.

## 6. Surrogates / ML at a 50–150-solve budget

The one angle where *no* claim survived to verification in the first pass —
so treat all of this as **[extracted]** and calibrate with the one strong
negative result:

* **DTU critical review** (Woldseth, Aage, Sigmund et al., SMO 2022,
  <https://arxiv.org/abs/2208.02563>): the NN-for-TO literature has produced
  *"few real breakthroughs"*; direct design-prediction nets have *"varying
  success"*; reported speed-ups are measured in SIMP iterations on cheap
  linear solves — a regime that does not transfer here. **Direct-prediction
  NN TO: no-go.**
* **MFSE + Kriging** (Luo et al., CMAME 2020,
  <https://www.sciencedirect.com/science/article/abs/pii/S0045782520301493>):
  *material-field series expansion* compresses the topology to **≤50
  correlated coefficients** (mesh-independent), then a sequential Kriging
  loop with two infill criteria optimises using **only scalar objective
  values** — demonstrated on hyperelastic (nonlinear, no-easy-adjoint)
  problems. A DNN variant (2022,
  <https://www.sciencedirect.com/science/article/pii/S026412752200507X>)
  handles ~200 coefficients.
* **SOLO** (self-directed online learning, Nature Comms 2021,
  <https://www.nature.com/articles/s41467-021-27713-7>): online DNN surrogate
  + dynamic sampling around the predicted optimum; a compliance benchmark
  converged in **286 true FE evaluations** — 2–6× above oropt's budget, right
  order of magnitude.

**Why this still matters:** it attacks the actual bottleneck (number of
13-min solves) instead of the update rule, needs *no field sensitivity at
all*, and oropt uniquely owns a validated **35× cheaper proxy**
(`fastmode.py`, tied-linear, ~14 % stress bias) that could pay for the
surrogate's sample appetite — e.g. Kriging over MFSE coefficients evaluated
on fast mode, with periodic full nonlinear solves as the high-fidelity
correction (a classic multi-fidelity co-Kriging setup). The MFSE basis
(eigendecomposition of a spatial correlation over 575k centroids) needs a
Nyström/landmark approximation but is offline, once. **Prototype in phase 2**
— highest potential payoff, highest uncertainty; the open question from
verification stands: nobody has shown this working at exactly this budget on
plasticity+contact.

## 7. Nonlinear physics & robustness — mostly reassurance **[verified]**

* BESO-family hard-kill has been peer-reviewed-extended to combined geometric
  nonlinearity + elasto-plasticity (Habashneh & Movahedi Rad, Sci. Rep. 2022,
  <https://www.nature.com/articles/s41598-022-09612-z>) — small 2D benchmarks,
  but it confirms the seam's physics regime is publishable practice, not a
  hack.
* The same paper couples **reliability-based TO to BESO derivative-free**:
  reliability enters as a constraint on the volume fraction evaluated by
  Monte-Carlo (no RIA/PMA gradients, no extra FE solves when the random
  variable is the volume fraction). A cheap bolt-on if RBTO is ever wanted;
  not a priority.
* **ESL/DiESL** (Triller et al., SMO 2022,
  <https://link.springer.com/article/10.1007/s00158-022-03309-7>): the
  established adjoint-free route for crash-type TO — replace the nonlinear
  solve with linear static subproblems on deformed geometry. **Blocked for
  oropt today:** the only published DiESL topology variant runs **SIMP on the
  linear equivalent models**, i.e. it needs exactly the per-element modulus
  the SIMP spike showed OpenRadioss doesn't have. Revisit only with a
  standalone linear FE layer; an open research direction (flagged by the
  verification pass) is a DiESL outer loop with a *binary* (TOBS/SAIP) inner
  loop, which would dodge the modulus requirement — nobody has published
  that. Known theory caveat: ESLM can miss the true dynamic optimum (Stolpe
  2018).

## 8. What was searched and found nothing actionable

MMC/MMV moving morphable components (needs its own FE mapping / sensitivities,
fights the fixed-mesh deletion seam), Tosca controller internals (public docs
too thin to port from), and everything in the refuted list above. Coverage
caveat from the harness: only the top 25 of 93 extracted claims got
verification panels (budget), which is why §3, §5 and §6 carry
**[extracted]** markers — their primary sources were fetched and quoted, but
not adversarially cross-checked.

## 9. Implementation plan — status

1. **`saip.py` — SAIP optimiser** (`optimizer: saip`) — **IMPLEMENTED**.
   Binary contract identical to `tobs.py`; canonical-relaxation subproblem
   (analytic dual bisection on the value density `s_e/vol_e`, move-limited)
   plus an *oscillation damping* standing in for SCIP's conservatism (a
   rank-down on just-flipped elements — documented in the module as NOT the
   paper's reciprocal-asymptote formulation). Hermetic tests in
   `tests/test_saip.py`.
2. **MHCA** (variable neighbourhood radius on the existing HCA) —
   **IMPLEMENTED** (`hca.radius_start` / `radius_iters` / `radius_steps`,
   cached per-radius filter matrices, resume-safe schedule position via the
   loop's `set_iteration` hook). Tests in `tests/test_hca.py`.
3. **`controller.py` — multipoint back-off** — **IMPLEMENTED** for the volume
   target (`backoff_mode: multipoint` on every optimiser block: local linear
   fit of violation(vf) over the last `multipoint_window` history points,
   step to the `utilization_target` crossing, gate-authority clamps,
   checkpointed history, verbatim gate fallback whenever the fit is
   unusable). Adaptive per-case *weights* (LS-TaSC's second global set)
   remain future work. Tests in `tests/test_controller.py`.
4. **`levelset.update_rule: rde`** — **IMPLEMENTED** (one implicit
   reaction–diffusion step per iteration on the random-walk graph Laplacian
   of the existing smoothing operator, solved by a fixed-point contraction;
   `diffusion` is the complexity knob; tau bisection / prune re-sync /
   checkpointing shared with the classic rule). Tests in
   `tests/test_levelset.py`.
5. **TDSA** — **offline prototype implemented** (`oropt/tdsa.py`): verified
   closed-form 2D/3D topological derivatives of compliance as
   stress-quadratic forms, energy-proxy comparison and rank-agreement
   helpers; the module docstring records the exact verification chain. The
   remaining step to a `sensitivity: tdsa` mode is stress-*tensor* extraction
   from the engine anim (today only von-Mises is read). Tests in
   `tests/test_tdsa.py`.
6. **MFSE + Kriging over `fastmode`** — **offline prototype implemented**
   (`oropt/mfse.py`): Nyström MFSE basis over element centroids, exact
   volume thresholding, numpy-only GP + expected-improvement loop,
   deterministic seeded runs. Go/no-go on a coarse proxy mesh (with
   `fastmode` as the evaluator) still precedes any loop integration. Tests
   in `tests/test_mfse.py`.

## Sources

Primary sources cited inline. Verification: 23 confirmed / 2 refuted of 25
panel-checked claims (refuted: PTO speed-at-expensive-FE, PTO
"sensitivity-free"); 104 research agents, July 2026.

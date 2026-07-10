# oropt roadmap — what to build next (2026 deep-research synthesis)

*Status: research/planning document. Nothing here is implemented yet — it is a
prioritised backlog. Companion to [`topology_sota_2026.md`](topology_sota_2026.md)
(the algorithm survey whose shortlist is already wired in) and the design notes in
`docs/`.*

*Method: five parallel deep-research passes over the live codebase — algorithms,
compute cost, engineering/code-quality, validation, and product/UX — each reading
the actual source and citing `file:line`. This document merges and cross-ranks
their findings. Claims that were spot-verified against the tree during synthesis
are marked **[verified]**.*

---

## 0. Where the project stands

`oropt` is a mature, well-instrumented tool: ~11.7k LOC in `oropt/`, five
production optimisers (BESO / level-set / TOBS / HCA / SAIP) sharing one
solve→rank→delete→re-solve loop, three offline research prototypes
(`simp.py`, `tdsa.py`, `mfse.py`), 752 hermetic tests, ruff-clean, with strong
observability (`run.log`, divergence monitor, null-solve guard) and rich
post-processing (report.html, d3plot, STL, evolution GIF).

The engine and the SOTA algorithm shortlist are **not** where the remaining work
is. The five research passes converge on the same picture: the biggest gains are
in **(a) closing the validation gap** (the tool has never been proven to produce a
verified-optimal design end-to-end), **(b) attacking the solve-count bottleneck**
with the multi-fidelity machinery that is 90 % built but not wired, and **(c)
removing first-hour adoption friction** (you cannot run *anything* without a full
OpenRadioss stack, and the one example config is stale and machine-locked).

---

## 1. Top cross-cutting priorities (do these first)

Ranked by value ÷ effort across *all* five dimensions. Each links to its detailed
section.

| # | Item | Dimension | Effort | Why it leads |
|---|------|-----------|:------:|--------------|
| **P1** | **Fix the one example config** — it is legacy-format + hard-coded `E:\`/`C:\` paths **[verified]** | Product | **S** | The only worked example teaches the *deprecated* schema and cannot run anywhere unmodified — every new user starts broken |
| **P2** | **`demo`/synthetic solver backend** — run the whole GUI+pipeline with zero OpenRadioss | Product | S–M | Single highest-leverage change for evaluation; also unlocks real-solver-free e2e demos |
| **P3** | **Golden-fixture tests for the real-solver output parsers** | Eng / Validation | S | `runner.py` regex-scrapes solver text with **zero test coverage** on the single most fragile seam; an OR version bump breaks every run silently |
| **P4** | **Stress-tensor extraction spike** — `/ANIM/BRICK/TENS/STRESS` is *already emitted* by fastmode **[verified]** | Algorithms | S | One-solve spike unblocks TDSA, stress-constrained TO, and fatigue *at once* — the survey mis-priced this as "solver-output work" |
| **P5** | **Adaptive per-load-case weights** — finish LS-TaSC's 2nd global set | Algorithms | S–M | All the data is already computed each iteration; fixes the documented multi-load path-collapse failure mode at zero extra solves |
| **P6** | **Tiny real-solver smoke deck + nightly Docker-MUMPS integration job** | Validation | S–M | Keystone: unblocks every benchmark, cross-optimiser comparison, and fastmode re-validation below |
| **P7** | **CI matrix + mypy + coverage** — CI is Windows-only, no type-check, no coverage | Eng | S | Code is already ~fully annotated; turning on mypy is cheap and Ubuntu is where the TetGen/path bugs hide |
| **P8** | **Multi-fidelity fast-mode schedule** — cheap linear ranking every iter, periodic full solve for feasibility | Performance | M | Largest *untapped* speedup (~4–35×); all plumbing exists, only the per-iteration mode selector is missing |

**Suggested first sprint:** P1 + P3 + P7 are each small, independent, and land
immediately. P2 and P6 are the two "unlock everything else" investments. P4 is a
one-day spike whose result decides a whole algorithm branch.

---

## 2. Validation & scientific credibility

*The weakest area relative to the code's maturity. Validation is a **two-point
anchor** (one reproduced baseline σ=308.305 MPa / disp=1.229 mm, one hand-deletion
that solved). Everything between "one solve is correct" and "a full run produces a
trustworthy design" is unverified against the real solver. The strongest evidence
this matters is `docs/levelset_stuck_analysis.md`: a run reached convergence-like
behaviour while **destroying** the design, caught only by forensic post-mortem
after ~17 h of solves.*

| # | Artifact to build | Closes | Effort |
|---|-------------------|--------|:------:|
| V1 | **Tiny real-solver smoke deck** (~10 TET4, loaded through one contact patch) + `@pytest.mark.integration` test gated on `OROPT_OR_ROOT`; **nightly/​`workflow_dispatch` Docker-MUMPS job** | Hermetic CI cannot catch any deck-generation / solver-seam regression | S–M |
| V2 | **Automated post-deletion physics sanity check** (`oropt/sanity.py`): new-cavity/​self-contact detection, thin-web/​severance measure, free-node-pinning audit — geometric, pre-solve, hermetic | Self-contact & severance are only "worth a look" (README) — no gate; the levelset collapse would have been caught | M |
| V3 | **Canonical benchmark with a literature-known optimum** (MBB beam / cantilever / L-bracket as TET4, `configs/benchmark_mbb.yaml`) → BESO to convergence, compare compliance/topology to published result, commit as golden regression | No end-to-end run ever validated against a known answer | M–L |
| V4 | **Cross-optimiser benchmark harness** (`scripts/benchmark_optimizers.py`): run all 5 on the V3 problem from an identical start, tabulate final mass/stress/compliance/iters/wall-time/determinism-hash + convergence overlay | 5 optimisers share a loop but are **never compared head-to-head** in-tool; every "BESO carves fast / level-set fairs boundaries" claim is from literature, not measured here | M |
| V5 | **Geometric manufacturing-constraint verifier** (`oropt/mfg_verify.py`): independently measure overhang angle, min member, undercut-free draw columns, extrusion cross-section constancy, symmetry residual on the *final* mask/STL; wire into `report.py` as a manufacturability audit | `manufacturing.py` tests check the code runs, not that output geometry satisfies the constraint; `keep_connected` runs *after* mfg and can break it | S–M |
| V6 | **Determinism test + reproducibility doc**: run the loop twice with a stubbed deterministic solver, assert byte-identical masks/history across all optimisers; per-optimiser resume-determinism test | Reproducibility is claimed (config snapshot) but never asserted; the production path *is* deterministic (`np.argsort(kind="stable")` everywhere; RNG only in `mfse.py`) so this is cheap | S |
| V7 | **Parameter-sweep / sensitivity tooling** (`scripts/sweep.py`) over (evolution_rate, filter_radius, history_weight) on the cheap benchmark; publish a robustness table in `docs/` | README calls BESO "heuristic, sensitive" but ships no sweep tool and no sensitivity study | M |
| V8 | **fastmode proxy re-validation harness**: solve fast + full for several designs at different volume fractions **and both load directions**, plot σ_fast/σ_nonlinear vs vf; commit the missing `FASTMODE_REPORT.md` data into `docs/` **[verified: report file is absent from the repo]** | The ~14 % stress bias is a *single* data point (one load case, full volume) extrapolated across a shrinking design that screens feasibility | M |

*Cross-cutting note: `validate.py` is the right architecture (gate errors before a
run) applied to the wrong scope — it covers **config**; V2 extends its philosophy
to **deck-generation + physics**. The Docker-MUMPS backend needs no Intel oneAPI,
so it is the natural substrate for the nightly integration runner that unblocks
V3/V4/V7/V8.*

---

## 3. Performance & compute cost

*The dominant cost is the FEA solve: ~13 min × 50–150 iters × N load cases =
11–33 h/run. There is no warm-start, no factorization reuse, no cross-iteration
state carry-over — each iteration solves from t=0.*

**Available now (config only):**

| # | Lever | Effort | Payoff | Where |
|---|-------|:------:|--------|-------|
| C1 | **`run.solver_concurrency > 1`** — parallel load cases (the `ThreadPoolExecutor` path already exists; default is 1) | S (built) | up to N× on multi-case runs | `loop.py:795-834`, `config.py:64` |
| C2 | **Docker MUMPS `np > 1`** — real domain parallelism (native is pinned np=1: SPMD+contact segfaults) | S | ~2–4× per solve | `runner.py:142-160`, `config.py:46` |
| C3 | **Back-off damping / `backoff_mode: multipoint` / `oscillation_damping`** — kill the feasibility-gate ping-pong, each bounce is a wasted 13-min solve | S | trims the oscillation tail (often 10–30 % of a run) | `controller.py`, `loop.py:1442-1466` |

**Biggest untapped win (needs build):**

| # | Lever | Effort | Payoff | Where |
|---|-------|:------:|--------|-------|
| C4 | **Multi-fidelity fast-mode schedule** — today `fast_mode` is a *per-case whole-run toggle*; make it a per-iteration selector: fast tied-linear (~35× cheaper) for ranking every iteration, full nonlinear every K iters (and near convergence) to anchor feasibility. Tie discovery, `build_fast_case`, and the identical `extract` path all already exist | M | ~4–5× overall (approaching 35× on screened iters) | `fastmode.py`, `loop.py:1170+`, `758-764` |
| C5 | **Seen-mask solve cache** — hash the per-case alive mask, short-circuit to the stored result on a repeat (oscillation tail & mfg-noop re-solves); generalises the existing `reuse_iter0_solve` guard | S–M | saves the repeat solves near convergence | `loop.py:694-725, 808-812` |
| C6 | **Implicit tolerance / timestep tuning** — `prepare_engine` copies solver controls verbatim; expose `/IMPL/*` knobs to trade Newton iters for speed | M | 1.2–2× per solve | `deck.py:454-479` |
| C7 | **Cross-config queue concurrency** — `queue_runner` runs configs strictly serially; cap-limited parallelism across independent already-isolated work dirs | M | K× *throughput* (batch) | `queue_runner.py:131-195` |

**Investigated, deprioritised:** OpenRadioss **restart warm-start** is a near
dead-end — element deletion changes the DOF set every iteration, so the prior
`.rst` is topologically invalid as a continuation (C-tier). **Coarse→fine
multi-resolution** has large potential but **no tooling exists** (growthmesh only
*adds* material); it needs a coarsener + cross-mesh field transfer + per-resolution
contact regeneration — treat as a separate project.

---

## 4. Algorithms & research frontier

*The SOTA shortlist (SAIP, MHCA, multipoint controller, RDE level-set) is done.
The frontier is now: finish the partially-built prototypes, and the few 2024–26
methods that fit the binary-mask / energy-only / expensive-solve seam.*

**Headline finding [verified]:** the stress **tensor is already requestable and
already written** — `fastmode.py:379` emits `/ANIM/BRICK/TENS/STRESS`, asserted by
`tests/test_fastmode.py`. The survey priced TDSA's blocker as "solver-output work";
in fact the deck card exists and flows through the same anim→pyvista path as
von-Mises. The only unverified link is whether `anim_to_vtk` surfaces the tensor as
a per-cell array (`results.py` reads only the scalar `F_VONMISES`, `results.py:149`)
— a **one-solve spike** (P4), not a research problem.

| Rank | Item | Effort | Risk | Payoff |
|:----:|------|:------:|:----:|--------|
| A1 | **Adaptive per-load-case weights** — LS-TaSC's 2nd global set, marked future work at `controller.py:44`. Per-case utilisation ratios are already computed (`loop.py:630-646`); the `MultipointBackoff` fit machinery generalises from `vf` to `weight_i`; weights feed one spot (`combine_sensitivity`, `beso.py:125`). Opt-in key mirroring `backoff_mode: multipoint` | S–M | Low | Med–High — fixes documented multi-load path collapse |
| A2 | **Stress-tensor → `Results.stress`** (P4 spike, then parse 6 components in `results.py` alongside `F_VONMISES`) — *enabling infra* for A4/A5/fatigue | S | Low–Med | High (enabling) |
| A3 | **Real SCIP** — replace SAIP's `oscillation_damping` stand-in (honest about not being the paper's formulation, `saip.py:37-47`) with reciprocal-variable MMA moving asymptotes. Stable *without* an active volume constraint — exactly oropt's moving-target regime. Degrades gracefully to today's SAIP | M | Low | Med |
| A4 | **`sensitivity: tdsa` mode** (needs A2) — feed `tdsa.td_compliance_3d` (already implemented & verified, `tdsa.py:247`) to all optimisers. Modest for compliance-min (energy already ranks similarly); the *prize* is the shared extraction | S–M | Med | Med |
| A5 | **Proper stress-constrained TO** (needs A2) — p-norm / augmented-Lagrangian stress *objective/constraint* instead of the scalar-peak feasibility gate + `addback_stress_bias` heuristic | M–L | Med | High (stress-driven parts) |
| A6 | **MFSE + multi-fidelity co-Kriging** — `mfse.py` is complete offline but single-fidelity; add a two-level Kennedy-O'Hagan GP (cheap = fastmode, expensive = full nonlinear). Attacks the real bottleneck (solve count) | L | High | High-or-zero — gate behind a coarse-mesh go/no-go |
| A7 | **DiESL with a binary inner loop** (survey's flagged *unpublished* open problem) — fastmode already *is* the linear engine; a binary inner loop dodges the SIMP modulus blocker. Needs new extraction (full nodal displacement + `K_lin·u`) | L | High | High (publishable) |
| A8 | **Eigenfrequency (NVH) constraint** — modal strain-energy is an energy-shaped field that fits the seam *if* the solver emits per-element modal energy (blocker is solver output, not the seam) | M–L | Med–High | Situational |
| A9 | **Fatigue/durability** (needs A2, niche), **buckling** (needs geometric-stiffness eigensolve, likely no-go today), **RBTO via MC on vf** (cheap bolt-on, niche) | S–L | — | Niche |
| — | **Multi-material / thermal — no-go.** Both break the 0/1-omit-card seam and `/SURF/PART/EXT` contact regeneration (same class of blocker as SIMP) | — | — | — |

**Sequencing:** A1 + the A2 spike now → if the tensor is exposed, wire A4 *and* A5
(shared extraction, A5 is the bigger prize) → A3 in parallel (self-contained) →
A6/A7 as phase-2 research bets gated by a coarse-mesh go/no-go.

---

## 5. Engineering & code quality

*The codebase is in notably good health: `from __future__ import annotations`
everywhere, pervasive return-type hints, no `TODO/FIXME/HACK`, no bare `except:`,
solver failures modelled as data (`RunResult`) not exceptions. Debt is concentrated
in a few god-functions, one large hand-rolled config schema, and one untested
high-risk seam.*

| # | Item | Effort | Value |
|---|------|:------:|-------|
| E1 | **Golden-fixture tests for the solver-output parsers** — `runner.py` `_starter_ok`/`_engine_ok`/`_parse_engine_stats`/`DivergenceMonitor` (`runner.py:179-395`) regex-scrape free text with **no test coverage**; capture 3–4 real `.out` snippets (success/error/diverge/truncated) and table-test | S | **High** (biggest robustness risk) |
| E2 | **CI matrix + type-check + coverage** — add `{windows, ubuntu} × {3.10, 3.12}` (pyproject claims `>=3.10`, only 3.12/Windows tested), `mypy oropt` (code is ~fully annotated → cheap), non-gating coverage, a `.pre-commit-config.yaml` running ruff | S–M | **High** (compounding) |
| E3 | **Make unknown config keys an error** — `from_dict` silently drops unrecognised keys; the `unknown_keys` walker only *warns* and only if the caller passes `raw=`. A typo like `evolution_ratte:` reverts to default and runs for hours | S | High (prevents silent multi-hour surprises) |
| E4 | **Decompose `run_optimization`** — a ~620-line god-function (`loop.py:940-1559`) owning resume/restore, setup, the iteration body, and finalisation; extract `_setup_design_space` / `_restore_checkpoint` / `_run_iteration` / `_finalize` to unlock direct unit tests of resume/convergence | M | High |
| E5 | **De-god the GUI / validate / report modules** — `render_constraints_tab` (~600 lines, `gui/app.py:479`), `check_config` (~308 lines, `validate.py:150`), `report.py` (summary+charts+3D+templating in one 1236-line file) | M (parallelizable) | Medium |
| E6 | **Config schema hardening** — 20 dataclasses / 267 fields with validation split across `__post_init__`, `unknown_keys`, and `check_config`; evaluate a pydantic v2 migration (folds coercion + `extra="forbid"` + ranges into the types). Large; weigh against "it works and is tested" | M–L | Medium |
| E7 | **Deck-parser robustness** — dual whitespace/fixed-width parsing (`deck.py:31-49`) raises raw `ValueError`/`IndexError` with no line context on a malformed row; centralise field parsing with line-numbered errors | M | Medium |
| E8 | **Packaging hygiene** — add cautious upper bounds on fast-moving deps (`pyvista`, `streamlit` pin only floors), a `py.typed` marker (missing, so downstream loses the hints), `dynamic = ["version"]` (dup between pyproject and `__init__`) | S | Low–Med |
| E9 | **Property-based tests (Hypothesis)** — deck round-trip invariants over arbitrary masks; growthmesh geometry sign/quality; the numeric kernels in `simp.py`/`tdsa.py`/`mfse.py` (density∈[0,1], mass-conserving filter, volume respected) | M | Medium |

---

## 6. Product, UX & packaging

*The engine, validation, observability, and post-processing are strong. The
adoption ceiling is set almost entirely by **first-hour friction**.*

| # | Item | Effort | Value |
|---|------|:------:|-------|
| U1 | **Fix `configs/elevator_linkage.yaml`** — it uses the **legacy** schema (`constraints:` block, `model.stem`, `disp_node_id`) and hard-codes `case_dir: E:\…` + `root: C:\OpenRadioss` **[verified]**. Rewrite in `load_cases:` format with relative/placeholder paths + inline comments; keep a separate `configs/legacy_single_case.yaml` as the migration fixture | S | **Highest** |
| U2 | **`demo`/synthetic solver backend** — an analytical proxy returning an energy-shaped field + stress/disp from the mesh, behind `backend: demo`, so the entire GUI/queue/monitor/report/GIF pipeline runs on a bundled toy deck with `pip install -e .[gui]` and nothing else | S–M | **Highest** |
| U3 | **All-in-one Docker image + `Dockerfile`/`docker-compose.yml` in-repo** — OpenRadioss-MUMPS + oropt + Streamlit → `docker run -p 8501:8501 oropt`. No in-repo Dockerfile exists today; the Docker path only dockerises the *solver* | M | High |
| U4 | **QUICKSTART.md + "which optimizer" decision table + troubleshooting** — no quickstart/tutorial/screenshots exist; the 751-line README is reference-grade but a brutal first read. A one-page problem-type→optimiser table (default→BESO; smooth boundary→level-set; ILP move-limit→TOBS; nonlinear-contact→HCA; fast analytic→SAIP) is very cheap, high value | S–M | High |
| U5 | **Config presets** (`configs/presets/`: coarse_proxy, final_highfidelity, multiload, additive, docker_mumps) + **`oropt init` scaffolder** + **JSON Schema** emitted from the dataclasses for editor autocomplete via `# yaml-language-server: $schema=` | M | High |
| U6 | **Run-history browser + multi-run compare view** in the GUI — Monitor/Re-postprocess require *typing a run-folder path* (`app.py:1628`); nothing enumerates past runs. For a sweep-driven workflow there is no way to overlay convergence traces or compare topologies (data already in `history.csv`) | M | Med–High |
| U7 | **Mass & cost summary** — add a `material:` section (`density`, optional `cost_per_kg`); report starts/final **mass in kg** not just `%` removed. No material model lives in config today (AlSi10Mg is prose only) | S–M | Medium |
| U8 | **Download buttons** (report.html, STL, GIF) in the GUI — only the re-animated GIF is downloadable; sharing otherwise needs filesystem access to the run host | S | Medium |
| U9 | **Fix README refresh-interval docs** — README says "default 60 s" but the code enforces `min_value=120, value=120` **[verified: `app.py:1912`]** | S | Low (correctness) |
| U10 | **STEP/solid CAD export** — STL/VTP are faceted surface meshes; engineers need STEP/IGES B-rep to return the design to CAD/PLM (via gmsh/OpenCASCADE, or at minimum a documented STL→STEP recipe) | L | Medium |
| U11 | **De-elevator the framing + "can I use this for my part?" checklist** — the tool is mechanically general (any OpenRadioss TET4 implicit deck) but framed around one part. State the baked-in assumptions honestly: TET4 solids only (no shell/hex), implicit nonlinear, `/ANIM/ELEM/ENER` required, compliance objective only | S | Medium |

---

## 7. What was checked and deliberately deprioritised

- **OpenRadioss restart warm-start** — invalid across element-deletion DOF changes (§3).
- **Coarse→fine multi-resolution** — high potential, no tooling, needs a separate project (§3).
- **Multi-material / thermal objectives** — architecturally blocked by the 0/1-omit-card seam (§4).
- **Direct-prediction neural-network TO** — the survey's documented no-go (DTU review: "few real breakthroughs").
- **Full end-to-end real-solver CI on every PR** — impractical (hours per solve); the *parser* half (E1) and a *nightly* smoke job (V1) are the cheap, high-value slices.

---

## 8. Consolidated first-quarter plan

1. **Week 1 (small, independent, immediate):** U1 (fix example config), U9 (README refresh), E1 (parser fixtures), E3 (unknown-key error), E7/E8 hygiene, V6 (determinism test), P7/E2 (CI matrix + mypy + coverage).
2. **Unlock-everything investments (parallel):** U2 (demo backend) and V1/P6 (smoke deck + nightly Docker job). P4/A2 spike (stress tensor) — one day, decides the A4/A5 branch.
3. **Then, data-driven:** A1 (adaptive weights), C4 (multi-fidelity schedule), V3→V4 (benchmark → cross-optimiser comparison), U3/U4 (Docker image + quickstart).
4. **Phase-2 research bets (gated by a coarse-mesh go/no-go):** A5 (stress-constrained), A6 (MFSE co-Kriging), A7 (DiESL-binary).

*Every validation item should follow the gold standard already in the repo:
`docs/levelset_stuck_analysis.md` + its era-pinned repro scripts — a synthetic case
driven through the exact production code chain, asserted bit-identical, with the
mechanism confirmed and alternatives explicitly falsified. The team's problem is not
that it cannot validate; it is that validation is currently **reactive**
(post-mortem) rather than **preventive** (a gate before/within the run).*

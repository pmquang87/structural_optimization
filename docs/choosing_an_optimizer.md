# Choosing an optimiser

oropt ships five discrete optimisers that plug into the same solve → rank →
delete → re-solve loop and are selected with a single `optimizer:` key. They all
share the `/ANIM/ELEM/ENER` energy sensitivity, the element-deletion deck path,
and the multi-load / manufacturing / connectivity machinery — **only the
per-iteration update differs**. Start with the default; reach for another when
its bias matches your problem.

## Decision table

| Your problem / goal | Use | `optimizer:` | Why |
|---|---|---|---|
| **Default — just optimise the part** | **BESO** | `beso` | The bi-directional evolutionary scheme: ranks elements by energy density, deletes the least useful with add-back. Fast, well-understood, reaches target volume in few iterations. It is heuristic (sensitive to `evolution_rate` / `filter_radius` / `history_weight`), so start conservative and watch the traces. |
| **Smoother, more manufacturable boundary** | **Level-set** | `levelset` | A nodal level-set: energy → nodal velocity → φ evolution + smoothing → bisected threshold. Yields smooth, low-stress-concentration interfaces instead of BESO's ragged element-by-element edge. Use `update_rule: rde` (reaction–diffusion) for an unconditionally-stable step with `diffusion` as the single complexity knob. Evolves slowly — best as a finishing stage. |
| **Formal move-limited binary optimisation** | **TOBS** | `tobs` | Topology Optimisation of Binary Structures: each iteration's element flips come from an integer linear program (`scipy` HiGHS) with a formal move limit (`flip_limit`) and a linearised, ε-relaxed volume constraint (`constraint_relaxation`) — a principled subproblem rather than a heuristic threshold. |
| **Nonlinear / contact robustness, no gradients** | **HCA** | `hca` | Hybrid Cellular Automata (the LS-TaSC method, built for exactly this gradient-free nonlinear/contact regime): a persistent per-element virtual density driven by a proportional controller toward an energy-density setpoint. Optional MHCA variable-neighbourhood schedule (`radius_start`) for global-then-local search. Good at settling a design that oscillates against a stress limit. |
| **Fast analytic flips at large mesh size** | **SAIP** | `saip` | Sequential Approximate Integer Programming (Liang & Cheng): the binary subproblem is solved *analytically* by the canonical relaxation — a per-element sign test with a bisected volume multiplier, no MILP solver, microseconds at any mesh size. `oscillation_damping` adds SCIP-inspired conservatism so add/remove ping-pong decays. |

## How to choose, in one paragraph

If you have no strong reason otherwise, use **BESO** — it is the default and the
best-trodden path. Switch to **level-set** when the *boundary quality* matters
(stress concentrations, manufacturable surfaces). Use **TOBS** when you want a
formal move-limited ILP step instead of a heuristic threshold and can afford the
solver. Use **HCA** when the model is strongly nonlinear / contact-heavy and BESO
fights the stress limit. Use **SAIP** when the mesh is large and you want
analytic, solver-free flips. All five obey the same feasibility gate / back-off
controller, manufacturing constraints, growth regions and multi-load aggregation.

## Staging: switch optimisers mid-run

You do not have to pick one for the whole run. The classic win is **an aggressive
discrete method to find the topology fast (BESO/TOBS), then a short level-set pass
to fair out the boundary**. A "switch" is just a `--resume` with a changed
`optimizer:` — the binary alive mask carries over; each optimiser's continuous
field is rebuilt from it. The big footgun: switching swaps the *entire* config
block at once (`target_volume_fraction`, `filter_radius`, `evolution_rate`, …), so
align the destination block first.

Full mechanics, advantages, risks and the recommended two-stage pattern:
[`docs/optimizer_switching.md`](optimizer_switching.md).

## See also

- The per-optimiser knob reference is in the README "Configuration highlights"
  section (each `<name>:` block).
- `docs/topology_sota_2026.md` — the research survey behind the algorithm
  portfolio.
- `docs/levelset_stuck_analysis.md` — level-set failure modes to watch for.

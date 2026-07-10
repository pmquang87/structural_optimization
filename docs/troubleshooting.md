# Troubleshooting

oropt is built to fail *loudly and early*: a bad config is caught in ~1 s before
launch (`oropt.validate`), and several run-start guards + in-loop guards abort with
a specific reason rather than silently producing a wrong design. This page maps
the failure modes the code already handles to their cause and fix.

**Where to look first, every time:**

- **`run.log`** in the run folder (`work_dir`, or `model.case_dir` when blank) —
  the full tee'd log of the run, including every best-effort post-run step's skip
  reason. A fresh run truncates it; a `--resume` appends.
- The **Monitor tab's "Run log (tail)"** panel surfaces the same log in the GUI.
- **`status.json`** — the `state` (`running` / `converged` / `stopped` / `failed`)
  and `message` fields carry the abort reason.
- The per-solve OpenRadioss listing `<stem>_0001.out` (engine) / `<stem>_0000.out`
  (starter) in the `solve/` sub-folder for raw solver errors.

## Symptom → cause → fix

| Symptom | Cause | Fix |
|---|---|---|
| **`SOLVE FAILED (null solve): … produced zero stress and zero strain energy`** — run stops `failed`, often at iteration 0. | The solve reached NORMAL TERMINATION but the model carried **no load**: the force landed on a constrained / rigid DOF, a contact never engaged, or the deck was mis-exported. Every feasibility metric reads 0 and the energy sensitivity is uniformly zero, so oropt refuses to "optimise" a dead model (loop null-solve guard). | Fix the load path in the deck — verify the `/CLOAD` (or imposed motion) acts on free design DOFs and the contact engages. Re-solve the base deck in OpenRadioss standalone and confirm non-zero stress before re-running. |
| **`case … did not converge -- treated as INFEASIBLE, backing off (n/N consecutive)`**, then eventually **`n consecutive iterations did not converge`** → `failed`. | The implicit engine hit a **divergence streak** (`--ITERATION DIVERGE` / collapsing timestep with no accepted step) — usually an over-carved design that severed the load path. The watchdog (`run.diverge_max_cycles`) kills the solve, the loop backs the volume target off and re-grows from the previous mask; `run.diverge_fail_after` consecutive non-converged iterations fail the run. | Remove material more gently: lower `evolution_rate`, raise `target_volume_fraction`, or increase `protect_layers` / `contact_protect_dist`. Enable the feasibility back-off (`backoff_gain` 10–20, `damping_threshold` ~0.95) so it eases into the limit. If healthy solves are being cut, raise `run.diverge_max_cycles`. |
| **`error: model.case_dir does not exist` / `load case … starter (or engine) deck not found`** — run aborts before launch (exit code 2). | The deck folder or a `<stem>_0000.rad` / `_0001.rad` pair is missing: wrong `model.case_dir`, a wrong load-case `stem`, or the decks were never placed there. | Point `model.case_dir` at the real folder and make each load case's `stem` match its actual deck files. Validate first (see QUICKSTART step 2) so this is caught in 1 s, not after a solve. |
| **Run aborts at start naming the deck's real `/GRNOD/NODE` ids** — a configured group id "selects zero nodes" / "is not in the deck". | A **typo'd or absent group id**: `model.bc_group_id`, a `freeze_group_ids` entry, or a `stress_exclude_group_ids` entry that the deck does not contain (or a group that lists no nodes). `loop.validate_group_ids` fails fast so a mistyped id can't silently disable its whole region hours in. | Correct the id to one the error message lists. The GUI's **🔍 Preview region element counts** button runs the same guard without starting a run. |
| **`growth box '…' contains no design elements`** (or the preview reports 0 candidates) — abort at run start. | An **empty growth region**: the box/sphere/cylinder/polyhedron sits over space that has no candidate mesh — either coordinates outside the meshed design part, or a region that was never pre-meshed. | Fix the region coordinates so they cover meshed design elements, or generate candidate mesh with the growth-mesh PREPARE step (`python -m oropt.growthmesh --config …` / the GUI's **⚙️ Generate growth mesh**) and point `case_dir` at the extended decks. Preview before committing. |
| **Status stuck on `running` but nothing is happening** (no new `history.csv` rows, `run.log` not growing) — typically after a crash or a cluster/session kill. | The process was hard-killed mid-solve, so `run.py`'s `_mark_failed` (which stamps `failed` and clears the pid on an unhandled exception) never ran and the terminal `running` status was left behind. | In the GUI use **⏹ Force kill** to clear the stale state, then **↻ Resume** to continue from the last checkpoint. For scheduled/cluster runs, set `run.max_wall_hours` below your session limit so the run stops **cleanly** (`stopped`, checkpoint kept, post-run steps still run) instead of being killed mid-solve. Inspect `run.log` for the last activity before the gap. |

## General tips

- **Validate before every run.** `python -m oropt.run` runs the check
  automatically and aborts on errors; add `--skip-validate` only deliberately.
  Warnings (including *unrecognised config key* — a typo like `evolution_ratte:`
  that would silently revert to default) are printed but don't block.
- **Develop on a coarse proxy** (`configs/presets/coarse_proxy.yaml`) to surface
  path / group-id / load-path problems in minutes, not after a 13-min solve.
- **BESO is heuristic** — if the design collapses or oscillates, start
  conservative (small `evolution_rate`, moderate `filter_radius`) and watch the
  mass / σ / displacement traces in `report.html` or the Monitor.
- Best-effort post-run steps (d3plot / smoothing / animation / report) never fail
  the run; if one is missing, its skip reason is in `run.log`.

See also: [`docs/applicability.md`](applicability.md) (is my part supported?),
[`docs/choosing_an_optimizer.md`](choosing_an_optimizer.md), and
[`docs/levelset_stuck_analysis.md`](levelset_stuck_analysis.md) (a forensic
level-set failure walkthrough).

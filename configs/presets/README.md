# Config presets

Ready-to-tweak starting points for common oropt workflows, all derived from the
reference example (`../elevator_linkage.yaml`) and written in the current
`load_cases:` schema. Copy one, point `model.case_dir` at your own deck folder,
adjust the load-case `stem`s / limits, and run:

```
python -m oropt.run --config configs/presets/<preset>.yaml
```

Every preset loads and passes `oropt.validate.check_config` (the only errors on a
fresh checkout are the expected "path/deck/executable not found" ones, because
the paths are placeholders).

| Preset | Use it when… | Key differences from the reference |
|---|---|---|
| **coarse_proxy.yaml** | Dialling in the setup on a coarse proxy mesh — fast, disposable. | Aggressive `evolution_rate: 0.04`, short `max_iter: 20`, no archiving. |
| **final_highfidelity.yaml** | Producing a deliverable design. | Conservative `evolution_rate: 0.01`, full `max_iter: 150`, feasibility back-off on (`backoff_gain`, `damping_threshold`), full archiving (`archive_iterations` + `archive_restart`). |
| **multiload.yaml** | Optimising one structure against several loads. | Two `load_cases` with `weight`s, each with its own `sigma_allow` / displacement limits. |
| **additive.yaml** | The part is powder-bed-fusion printed (AlSi10Mg). | `manufacturing` constraints ON: `min_member_layers`, `build_direction`, `max_overhang_angle`. |
| **docker_mumps.yaml** | No native OpenRadioss — solve via the Dockerised MUMPS build. | `docker.enabled: true` with `np > 1` (real MPI); `or_paths` / Intel-MPI `run` settings are ignored. |

Notes:

- Archiving is disk-heavy (see the README "Disk cost" note): `final_highfidelity`
  turns it fully on for replayability; the others leave it off.
- `docker_mumps` is the only preset that runs without a native OpenRadioss +
  Intel oneAPI install; the rest assume the native Windows backend (toggle
  `docker.enabled: true` in any of them to switch).
- Pick an optimiser with `docs/choosing_an_optimizer.md`; all presets default to
  `beso`.

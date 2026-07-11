# The oropt Docker image

An in-repo `Dockerfile` + `docker-compose.yml` (roadmap item U3, the shippable
slice) that packages **oropt + the Streamlit GUI** into a container:

```
docker build -t oropt .        # or just:
docker compose up              # builds on first run, then serves the GUI
```

then open <http://localhost:8501>.

**What the image contains:** Python 3.12, `oropt` with the `[gui]` extra
(streamlit, pyvista/VTK with the headless OS libraries), a non-root `oropt`
user, a healthcheck on the Streamlit port, and a `/cases` volume mount point
for your decks, configs and run output.

**What it does not contain: OpenRadioss.** The MUMPS-implicit solver image
(`openradioss-mumps:*`) is **not publicly distributed with this repo** — you
build or load it separately (see the image's own `COLLEAGUE_INSTRUCTIONS.md`).
The oropt image therefore cannot bundle the solver by default; pick the usage
mode below that matches what you have.

## Do you even need this image?

If you run oropt **natively** — Windows + native OpenRadioss/Intel-MPI, or any
host where `pip install -e .[gui]` works — you do **not** need this image at
all. `streamlit run oropt/gui/app.py` on the host is the simpler, fully
supported path (see [QUICKSTART.md](../QUICKSTART.md)). The image exists for
people who want a no-Python-setup evaluation or an all-container deployment.

## Usage modes

### (a) Demo backend — zero-solver evaluation

The `demo` synthetic solver backend (`demo.enabled: true` in the config)
returns an analytical energy-shaped field from the mesh, so the entire
GUI → queue → optimiser → monitor → report → GIF pipeline runs with **no
OpenRadioss anywhere**. This is the mode the stock image fully supports on its
own and the recommended first contact:

```
docker compose up
# open http://localhost:8501, pick/build a config with demo.enabled: true
```

Put decks/configs under `./cases` on the host (mounted at `/cases` in the
container) and use `/cases/...` paths in the config (`model.case_dir`,
`work_dir`). Everything the run writes (reports, VTU snapshots, STL, GIFs)
lands back in `./cases` on the host.

### (b) Native solver on the host — don't use this image

If you have a native OpenRadioss install, the container cannot reach it (the
GUI would need the host's executables, Intel-MPI environment and paths).
Install oropt on the host instead:

```
python -m pip install -e .[gui]
streamlit run oropt/gui/app.py
```

### (c) Dockerised MUMPS solver — docker-out-of-docker

With `docker.enabled: true`, oropt launches the solver by shelling out to the
`docker` CLI (`docker run --rm --shm-size=… -v <run_dir>:/data
openradioss-mumps:…`). When oropt itself is in a container, that requires
**docker-out-of-docker**: the host daemon's socket mounted into the GUI
container, which then launches solver containers as *siblings* on the host.

Setup:

1. Build/load the solver image on the **host** (it is user-supplied, not
   pulled): `docker load -i openradioss-mumps-20260520.tar` or your own build.
2. Rebuild the oropt image with the docker CLI included:
   `docker build --build-arg INSTALL_DOCKER_CLI=1 -t oropt .`
3. In `docker-compose.yml`, uncomment the
   `/var/run/docker.sock:/var/run/docker.sock` mount.
4. **Make the cases mount path-identical on both sides.** Because solver
   containers are started by the *host* daemon, the `-v <run_dir>:/data` bind
   mount oropt issues is resolved against **host** paths. Replace
   `./cases:/cases` with an absolute same-path mount, e.g.
   `/home/me/oropt-cases:/home/me/oropt-cases`, and use that absolute path in
   your config — otherwise the solver container mounts an empty/nonexistent
   host directory.
5. Sort out socket permissions: the container runs as non-root `oropt`, so
   either add the host's docker group gid (`group_add: ["<gid>"]` in the
   compose service) or run the service as root.

Tradeoffs, stated plainly: mounting the docker socket grants the container
**effectively root on the host**; only do this on a machine whose GUI users
you trust. If that is unacceptable, run oropt natively on the host (mode b
plus `docker.enabled: true` — the recommended way to use the dockerised
solver) and skip this image.

## Licensing note: TetGen / growthmesh

The image deliberately omits the `[growthmesh]` extra: TetGen (used only by
the growth-mesh PREPARE step, `oropt.growthmesh`) is **AGPL-licensed**, and
baking it into an image you redistribute has licence implications the core
package avoids. If you need PREPARE inside the container, extend the image:

```dockerfile
FROM oropt
RUN pip install --no-cache-dir tetgen>=0.8.4
```

and evaluate the AGPL terms for your distribution before publishing the
result. `[dev]` (pytest/ruff/mypy) is likewise omitted — this is a runtime
image.

## Image details

- **Base:** `python:3.12-slim`, multi-stage (wheels built in stage 1; the
  runtime stage installs offline from those wheels — no compilers or pip cache
  in the final image).
- **Headless rendering:** `libgl1`, `libx11-6`, `libxrender1`, `libgomp1` +
  `PYVISTA_OFF_SCREEN=true`, so VTK/pyvista renders off-screen with no X
  server.
- **User:** non-root `oropt` (uid 1000).
- **Port:** 8501 (`--server.address 0.0.0.0`).
- **Healthcheck:** polls Streamlit's `/_stcore/health` every 30 s.
- **Build args:** `INSTALL_DOCKER_CLI=1` adds a static docker CLI (mode c
  only); `BASE_IMAGE=<image>` is an *experimental* hook to rebase onto a
  locally built solver image **if** that image ships Python ≥ 3.10 + pip —
  unverified, and mode (c) remains the supported way to combine GUI and
  solver.
- **Volumes:** mount your working area at `/cases` (compose does
  `./cases:/cases` by default; see mode (c) for the path-identity caveat).

## Troubleshooting

- *GUI up but blank/erroring visuals* → check the container logs for VTK
  library errors; the four OS libs above must be present (they are in the
  stock image — this matters if you changed `BASE_IMAGE`).
- *`docker CLI not found` in the GUI* → you enabled `docker.enabled: true`
  inside the container without rebuilding with `INSTALL_DOCKER_CLI=1`.
- *Solver container starts but finds no deck* → the mode (c) path-identity
  caveat: host daemon resolves bind mounts against host paths.
- General run issues → [troubleshooting.md](troubleshooting.md).

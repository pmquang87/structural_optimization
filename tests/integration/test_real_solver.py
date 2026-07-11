"""Real-solver integration smoke tests (roadmap item V1).

These tests run the tiny 20-TET4 fixture deck (``fixtures/smoke_0000.rad`` /
``smoke_0001.rad``) against a REAL OpenRadioss, exercising the solver seam the
hermetic suite can only mock: starter acceptance of oropt-written decks, the
engine's implicit solve, and the anim -> VTK -> Results extraction chain.

Gated — skipped unless the environment provides a solver:

* ``OROPT_OR_ROOT``      -> native backend: an OpenRadioss install root whose
  ``exec/`` holds ``starter_*`` / ``engine_*`` / ``anim_to_vtk_*`` binaries
  (Windows or Linux builds alike; non-MPI engine preferred, ``np=1``).
* ``OROPT_DOCKER_IMAGE`` -> docker backend: the Dockerised MUMPS-implicit build
  (see :class:`oropt.config.DockerOpts`). Extraction shells ``anim_to_vtk``
  through the same image via a generated wrapper script (POSIX-only).

With neither variable set the whole module skips instantly, so it is harmless
inside the default ``pytest`` run. Every solve is bounded by the config
timeouts (starter 300 s, engine 900 s hard / 600 s soft) — the fixture solves
in seconds on any working install, so a timeout means the backend is broken,
not slow.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from oropt.config import Config, LoadCase
from oropt.deck import Deck, prepare_engine
from oropt.results import extract
from oropt.runner import run_solver

FIXTURES = Path(__file__).parent / "fixtures"
STEM = "smoke"
DESIGN_PART_ID = 60000000
DESIGN_NODE_MIN = 60000000
BC_GROUP_ID = 60000000       # clamped z=0 face
DISP_NODE_ID = 60000015      # a loaded top-face node whose |disp| is tracked

# Generous-but-bounded solve budgets (seconds). The 20-tet fixture solves in
# seconds; these only exist so a wedged backend cannot hang the nightly.
STARTER_TIMEOUT_S = 300.0
ENGINE_TIMEOUT_S = 900.0     # hard kill
ENGINE_SOFT_TIMEOUT_S = 600.0  # judged non-converging past this

OR_ROOT = os.environ.get("OROPT_OR_ROOT", "").strip()
DOCKER_IMAGE = os.environ.get("OROPT_DOCKER_IMAGE", "").strip()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (OR_ROOT or DOCKER_IMAGE),
        reason="real-solver integration test: set OROPT_OR_ROOT (native "
               "OpenRadioss install root) or OROPT_DOCKER_IMAGE (Dockerised "
               "MUMPS build image tag) to enable"),
]


def _find_tool(root: Path, name: str, prefer_no_mpi: bool = False) -> str:
    """Root-relative path of an ``exec/<name>_*`` binary under a native install.

    Tolerates both Windows (``starter_win64.exe``) and Linux
    (``starter_linux64_gf``) build names. With *prefer_no_mpi* the pure-OpenMP
    variant wins over ``*impi*``/``*ompi*`` builds — the test runs ``np=1``
    without an MPI runtime.
    """
    cands = sorted(p for p in (root / "exec").glob(f"{name}_*") if p.is_file())
    if not cands:
        pytest.fail(f"OROPT_OR_ROOT={root} is set but no {name}_* executable "
                    f"exists under {root / 'exec'}")
    if prefer_no_mpi:
        cands.sort(key=lambda p: ("impi" in p.name.lower()
                                  or "ompi" in p.name.lower(), p.name))
    return f"exec/{cands[0].name}"


def _docker_anim_to_vtk_wrapper(case_dir: Path) -> str:
    """Write a wrapper that runs the image's ``anim_to_vtk`` on a host anim file.

    :func:`oropt.results.extract` invokes the native ``or_paths.anim_to_vtk``
    executable with the anim file path and captures stdout as the VTK; on the
    docker backend the equivalent tool lives inside the image, so this shim
    bind-mounts the anim's folder and forwards stdout. POSIX shell only — the
    docker leg targets the ubuntu nightly runner.
    """
    if os.name == "nt":
        pytest.skip("the docker-backend anim_to_vtk wrapper is POSIX-only; "
                    "on Windows use the native gate (OROPT_OR_ROOT) instead")
    wrapper = case_dir / "anim_to_vtk_docker.sh"
    wrapper.write_text(
        "#!/bin/sh\n"
        "# oropt integration shim: run the solver image's anim_to_vtk on \"$1\"\n"
        "# (host path); the VTK is written to stdout like the native tool.\n"
        "set -eu\n"
        'd=$(cd "$(dirname "$1")" && pwd)\n'
        f'exec docker run --rm -v "$d":/data -w /data {DOCKER_IMAGE} '
        'anim_to_vtk "$(basename "$1")"\n',
        encoding="utf-8", newline="\n")
    wrapper.chmod(0o755)
    return wrapper.name


def _make_config(case_dir: Path) -> Config:
    """A Config pointing at the copied fixture deck, backend picked by env."""
    cfg = Config()
    cfg.model.case_dir = str(case_dir)
    cfg.model.design_part_id = DESIGN_PART_ID
    cfg.model.design_node_min = DESIGN_NODE_MIN
    cfg.model.bc_group_id = BC_GROUP_ID
    cfg.load_cases = [LoadCase(
        name="smoke", stem=STEM, weight=1.0, sigma_allow=10000.0,
        disp_constraints=[{"node_id": DISP_NODE_ID, "d_allow": None}])]
    cfg.work_dir = str(case_dir)
    cfg.run.starter_timeout_s = STARTER_TIMEOUT_S
    cfg.run.engine_timeout_s = ENGINE_TIMEOUT_S
    cfg.run.engine_soft_timeout_s = ENGINE_SOFT_TIMEOUT_S
    cfg.run.np = 1
    cfg.run.nt = 2
    cfg.run.anim_dt = 1.0
    if DOCKER_IMAGE:
        cfg.docker.enabled = True
        cfg.docker.image = DOCKER_IMAGE
        cfg.docker.np = 1            # 20 elements: one MPI domain is plenty
        cfg.docker.nt = 2
        cfg.docker.shm_size = "1g"
        cfg.run.use_mpi = False
        # extract() runs or_paths.anim_to_vtk natively -> shim through the image
        cfg.or_paths.root = str(case_dir)
        cfg.or_paths.anim_to_vtk = _docker_anim_to_vtk_wrapper(case_dir)
    else:
        root = Path(OR_ROOT)
        cfg.docker.enabled = False
        cfg.run.use_mpi = False      # np=1 pure-OpenMP engine, no mpiexec needed
        cfg.or_paths.root = str(root)
        cfg.or_paths.starter = _find_tool(root, "starter")
        cfg.or_paths.engine = _find_tool(root, "engine", prefer_no_mpi=True)
        cfg.or_paths.anim_to_vtk = _find_tool(root, "anim_to_vtk")
    return cfg


@pytest.fixture
def case(tmp_path: Path) -> tuple[Config, Path]:
    """The fixture deck pair copied into an isolated dir + a matching Config."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    for suffix in ("_0000.rad", "_0001.rad"):
        shutil.copy(FIXTURES / f"{STEM}{suffix}", case_dir / f"{STEM}{suffix}")
    return _make_config(case_dir), case_dir


def _deletion_mask(deck: Deck) -> np.ndarray:
    """Alive mask with the ~20% lowest-id design elements deleted.

    On the fixture that is exactly the 2-tet sacrificial fin (whose private
    apex node is orphaned -> exercises the free-node pinning path) plus two
    bottom-cube tets; the remaining mesh stays one connected, supported piece.
    """
    n = deck.n_design_elements
    k = max(1, round(0.2 * n))
    alive = np.ones(n, dtype=bool)
    alive[np.argsort(deck.elem_ids)[:k]] = False
    return alive


def _write_step(cfg: Config, case_dir: Path, solve_dir: Path) -> dict:
    """Deck.write the 20%-deleted design into *solve_dir* (the loop's own path:
    BC-group nodes excluded from pinning, engine copied via prepare_engine)."""
    deck = Deck.load(case_dir / f"{STEM}_0000.rad", DESIGN_PART_ID,
                     DESIGN_NODE_MIN)
    alive = _deletion_mask(deck)
    solve_dir.mkdir()
    no_pin = set(int(v) for v in deck.group_nodes(BC_GROUP_ID))
    stats = deck.write(solve_dir / f"{STEM}_0000.rad", alive, no_pin=no_pin)
    prepare_engine(case_dir / f"{STEM}_0001.rad", solve_dir / f"{STEM}_0001.rad",
                   anim_dt=cfg.run.anim_dt)
    return stats


def test_iteration0_solve_and_extract(case):
    """Iteration-0 solve of the pristine fixture deck reaches ok=True and the
    extraction chain yields a real (non-null) Results with finite fields."""
    cfg, case_dir = case
    res = run_solver(cfg, case_dir)
    assert res.ok, f"iteration-0 solve failed at stage {res.stage!r}: {res.message}"

    r = extract(cfg, case_dir, stem=STEM)
    assert not r.is_null_solve, ("solver terminated normally but the design "
                                 "part carried no load (null solve)")
    assert r.element_ids.size == 20
    assert math.isfinite(r.sigma_max) and r.sigma_max > 0.0
    tip = r.disps.get(DISP_NODE_ID, float("nan"))
    assert math.isfinite(tip) and tip > 0.0
    assert np.all(np.isfinite(r.energy)) and np.all(np.isfinite(r.vonmises))
    # Print the scalars so the first real runs can freeze them as goldens.
    print(f"[integration golden] iter0: sigma_max={r.sigma_max:.6g} MPa "
          f"disp[{DISP_NODE_ID}]={tip:.6g} mm n_elem={r.element_ids.size} "
          f"cycles={res.cycles} elapsed_s={res.elapsed_s}")


def test_deletion_step_resolves(case):
    """One BESO-like step — delete the ~20% lowest-id elements via Deck.write
    (orphaning the fin apex, which must be pinned) — still solves ok=True."""
    cfg, case_dir = case
    stats = _write_step(cfg, case_dir, case_dir / "iter_0001")
    assert stats["elements_alive"] == 16
    assert stats["free_nodes_pinned"] >= 1, \
        "deleting the fin was expected to orphan (and pin) its apex node"

    res = run_solver(cfg, case_dir / "iter_0001")
    assert res.ok, (f"re-solve after deletion failed at stage {res.stage!r}: "
                    f"{res.message} (free-node pinning stats: {stats})")

    r = extract(cfg, case_dir / "iter_0001", stem=STEM)
    assert not r.is_null_solve
    assert r.element_ids.size == 16
    assert math.isfinite(r.sigma_max) and r.sigma_max > 0.0
    print(f"[integration golden] step1: sigma_max={r.sigma_max:.6g} MPa "
          f"disp[{DISP_NODE_ID}]={r.disps.get(DISP_NODE_ID)} mm "
          f"pinned={stats['free_nodes_pinned']}")


def test_written_deck_round_trip_accepted_by_starter(case):
    """The deck oropt writes (element removal + injected free-node /GRNOD +
    /BCS) is accepted by the real starter: 0 ERROR(S).

    A subset of test_deletion_step_resolves, kept separate so a starter-side
    rejection of the round-tripped deck is diagnosed apart from an engine-side
    convergence failure.
    """
    cfg, case_dir = case
    _write_step(cfg, case_dir, case_dir / "iter_rt")
    res = run_solver(cfg, case_dir / "iter_rt")
    if not res.ok:
        assert res.stage == "engine", \
            f"starter rejected the oropt-written deck ({res.stage}): {res.message}"
        pytest.fail(f"starter accepted the deck but the engine failed: {res.message}")

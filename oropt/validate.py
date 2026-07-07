"""Fast, side-effect-free validation of a :class:`~oropt.config.Config`.

A bad config used to surface only deep into a run -- a missing deck after the
~13-min starter+engine solve, a nonsensical volume target hours into an 11-33 h
loop. :func:`validate_config` catches those in ~1 s *before* anything launches,
classifying each problem as a hard ``error`` (the run cannot or must not start)
or a soft ``warning`` (it will run, but probably not as intended).

Wired into both entry points: ``python -m oropt.run`` prints the problems and
exits non-zero on any error before the loop starts, and the Streamlit GUI shows
them and blocks its Start button on errors. The backend-executable checks are
shared verbatim with :func:`oropt.runner.run_solver` (see
:func:`oropt.runner.backend_problems`) so the fast check and the real run agree.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import Config, unknown_keys
from .deck import read_solid_geometry
from .runner import backend_problems

ERROR = "error"
WARNING = "warning"

VALID_OPTIMIZERS = ("beso", "levelset", "tobs", "hca")
VALID_GROWTH_SHAPES = ("box", "cylinder", "sphere", "polyhedron")


def _is_vec3(v) -> bool:
    """True for a 3-number ``[x, y, z]`` list/tuple (a local-frame vector)."""
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def _parallel(a, b) -> bool:
    """True when 3-vectors *a* and *b* are (near-)parallel — their cross product
    is negligible, so they cannot span the box's local xy plane."""
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    scale = max(abs(x) for x in a) * max(abs(x) for x in b)
    return max(abs(cx), abs(cy), abs(cz)) <= 1e-9 * max(scale, 1.0)


def _check_polyhedron_points(b, label, err) -> None:
    """Validate a polyhedron region's explicit node set: a list of >= 4 points,
    every point all three coordinates as finite numbers (no defaults, no
    inference), spanning a non-degenerate (positive-volume) convex hull."""
    pts = b.points
    if not isinstance(pts, (list, tuple)) or len(pts) == 0:
        err(f"growth box {label!r}: shape 'polyhedron' needs a points list "
            "[[x, y, z], ...] giving every node's coordinates explicitly")
        return
    for i, p in enumerate(pts):
        if not _is_vec3(p) or not all(math.isfinite(float(v)) for v in p):
            err(f"growth box {label!r}: polyhedron point #{i + 1} must be 3 "
                f"finite numbers [x, y, z]: got {p!r}")
            return
    if len(pts) < 4:
        err(f"growth box {label!r}: a polyhedron needs at least 4 points to "
            f"enclose a volume: got {len(pts)}")
        return
    # scipy locally: only paid when a polyhedron region is actually configured
    from scipy.spatial import ConvexHull, QhullError
    try:
        ConvexHull(pts)
    except QhullError:
        err(f"growth box {label!r}: the polyhedron points are degenerate "
            "(coplanar or duplicated) -- their convex hull encloses no volume, "
            "so the region would select no elements")


def _check_local_frame(b, label, err, warn) -> None:
    """Validate a growth box's optional oriented local frame (origin + x_axis +
    xy_axis). No-op when no frame field is set (a plain world-aligned box)."""
    if b.origin is None and b.x_axis is None and b.xy_axis is None:
        return
    for fld, val in (("origin", b.origin), ("x_axis", b.x_axis),
                     ("xy_axis", b.xy_axis)):
        if val is not None and not _is_vec3(val):
            err(f"growth box {label!r}: local-frame {fld} must be 3 numbers "
                f"[x, y, z]: got {val!r}")
    if b.x_axis is None or b.xy_axis is None:
        warn(f"growth box {label!r}: an oriented local frame needs both x_axis "
             "and xy_axis -- only one was given, so the frame is ignored and the "
             "box stays world-axis-aligned")
    elif _is_vec3(b.x_axis) and _is_vec3(b.xy_axis):
        if all(x == 0 for x in b.x_axis) or all(x == 0 for x in b.xy_axis):
            err(f"growth box {label!r}: local-frame x_axis and xy_axis must be "
                "non-zero vectors")
        elif _parallel(b.x_axis, b.xy_axis):
            err(f"growth box {label!r}: local-frame x_axis and xy_axis are "
                "parallel -- they cannot define the box's xy plane")


@dataclass(frozen=True)
class Problem:
    """One validation finding: its *severity* (``error``/``warning``) and message."""
    severity: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.message}"


def has_errors(problems: Iterable[Problem]) -> bool:
    """True if any problem is a hard error (the run must not launch)."""
    return any(p.severity == ERROR for p in problems)


def _creatable(path: Path) -> bool:
    """Whether *path* could be created with ``mkdir(parents=True)``.

    True iff the nearest already-existing ancestor is a directory (so a fresh
    sub-tree can be made under it); False if that ancestor is a file, or if no
    ancestor exists at all (e.g. a non-existent drive letter on Windows).
    """
    path = path.resolve()
    for anc in (path, *path.parents):
        if anc.exists():
            return anc.is_dir()
    return False


def _docker_image_present(cfg: Config) -> bool | None:
    """Best-effort check that the Docker image exists locally.

    Returns True/False, or ``None`` when it cannot be determined (docker CLI
    absent, timeout, any error) -- callers treat ``None`` as "cannot verify, stay
    quiet" so validation never blocks on a flaky/slow daemon.
    """
    exe = cfg.docker.docker_exe
    if shutil.which(exe) is None and not Path(exe).exists():
        return None
    try:
        cp = subprocess.run([exe, "image", "inspect", cfg.docker.image],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return cp.returncode == 0


def check_config(cfg: Config, *, raw: dict | None = None,
                 probe_docker_image: bool = False) -> list[Problem]:
    """Structured validation of *cfg* -- the engine behind :func:`validate_config`.

    Pure and fast (no solver, no file writes). The only optional side effect is a
    short ``docker image inspect`` when *probe_docker_image* is set and the Docker
    backend is selected; it is off by default so the check stays hermetic.

    Pass *raw* (the mapping the config was parsed from, e.g.
    :meth:`Config.read_yaml_dict`) to also warn about unrecognised keys that
    :meth:`Config.from_dict` silently dropped -- a typo'd or misplaced knob that
    would otherwise revert to its default unnoticed.
    """
    problems: list[Problem] = []

    def err(msg: str) -> None:
        problems.append(Problem(ERROR, msg))

    def warn(msg: str) -> None:
        problems.append(Problem(WARNING, msg))

    # --- unrecognised config keys (silently dropped by Config.from_dict) ---
    if raw is not None:
        for key in unknown_keys(raw):
            warn(f"unrecognised config key {key!r} -- ignored (typo or wrong "
                 "section?); its default is used")

    # --- optimiser selector ---
    if cfg.optimizer_name() not in VALID_OPTIMIZERS:
        err(f"optimizer must be one of {', '.join(VALID_OPTIMIZERS)} "
            f"(got {cfg.optimizer!r})")

    # --- model directory / decks ---
    m = cfg.model
    if not cfg.load_cases:
        err("no load cases defined -- define at least one load case (Load cases tab)")

    case_dir = Path(m.case_dir).resolve()
    if not case_dir.exists():
        err(f"model.case_dir does not exist: {case_dir}")
    elif not case_dir.is_dir():
        err(f"model.case_dir is not a directory: {case_dir}")

    cases = cfg.load_case_list()
    for c in cases:
        if not (c.stem or "").strip():
            err(f"load case {c.name!r}: deck stem is required")
            continue
        if not c.starter.exists():
            err(f"load case {c.name!r}: starter deck not found: {c.starter}")
        if not c.engine.exists():
            err(f"load case {c.name!r}: engine deck not found: {c.engine}")

    # --- run / output folder ---
    run_folder = Path(cfg.run_folder())
    if not _creatable(run_folder):
        err(f"run folder is not creatable: {run_folder.resolve()}")

    # --- solver backend (shared source of truth with runner.run_solver) ---
    for msg in backend_problems(cfg):
        err(msg)
    if cfg.docker.enabled:
        if probe_docker_image and _docker_image_present(cfg) is False:
            warn(f"docker image not found locally: {cfg.docker.image} "
                 "(it will be pulled on first run, or the run will fail)")
    elif cfg.run.np != 1:
        warn(f"run.np={cfg.run.np}: the native implicit + solid-contact solver "
             "requires np=1 (it segfaults otherwise)")
    # --- engine non-convergence watchdog ---
    if cfg.run.engine_soft_timeout_s > 0 \
            and cfg.run.engine_soft_timeout_s >= cfg.run.engine_timeout_s:
        warn(f"run.engine_soft_timeout_s={cfg.run.engine_soft_timeout_s:g} >= "
             f"engine_timeout_s={cfg.run.engine_timeout_s:g}: the hard kill "
             "fires first (and fails the run), so the soft budget never "
             "triggers the treat-as-infeasible back-off")
    if cfg.run.diverge_fail_after < 1:
        err(f"run.diverge_fail_after must be >= 1: got "
            f"{cfg.run.diverge_fail_after}")
    if cfg.run.diverge_max_cycles < 0:
        err(f"run.diverge_max_cycles must be >= 0 (0 = off): got "
            f"{cfg.run.diverge_max_cycles}")

    # --- numeric sanity (knobs of the active optimiser block) ---
    opt = cfg.active_opts()
    tvf = opt.target_volume_fraction
    if not (0 < tvf <= 1):
        err(f"target_volume_fraction must be in (0, 1]: got {tvf}")
    elif tvf == 1:
        warn("target_volume_fraction is 1.0 -- no material will be removed")
    if opt.evolution_rate <= 0:
        err(f"evolution_rate must be > 0: got {opt.evolution_rate}")
    if opt.filter_radius < 0:
        err(f"filter_radius must be >= 0: got {opt.filter_radius}")
    # feasibility back-off controller (0 gain / threshold 1.0 = classic gate)
    if opt.backoff_gain < 0:
        err(f"backoff_gain must be >= 0 (0 = classic binary gate): "
            f"got {opt.backoff_gain}")
    if opt.backoff_cap <= 0:
        err(f"backoff_cap must be > 0: got {opt.backoff_cap}")
    if opt.backoff_floor < 0:
        err(f"backoff_floor must be >= 0 (0 = purely proportional back-off): "
            f"got {opt.backoff_floor}")
    elif opt.backoff_floor > opt.backoff_cap:
        err(f"backoff_floor must be <= backoff_cap: "
            f"got {opt.backoff_floor} > {opt.backoff_cap}")
    if getattr(opt, "nucleation_rate", 0.0) < 0:
        err(f"nucleation_rate must be >= 0 (0 = interface-only evolution): "
            f"got {opt.nucleation_rate}")
    if not (0 < opt.damping_threshold <= 1):
        err(f"damping_threshold must be in (0, 1] (1.0 = off): "
            f"got {opt.damping_threshold}")
    if opt.addback_stress_bias < 0:
        err(f"addback_stress_bias must be >= 0 (0 = off): "
            f"got {opt.addback_stress_bias}")
    if opt.addback_stress_bias > 0 \
            and not any(c.sigma_allow is not None for c in cases):
        warn("addback_stress_bias is set but no load case sets a sigma_allow "
             "limit -- the stress-responsive add-back bias will never engage")
    any_disp_limit = any(dc.d_allow is not None
                         for c in cases for dc in c.disp_constraints)
    if (opt.backoff_gain > 0 or opt.damping_threshold < 1) \
            and not (any(c.sigma_allow is not None for c in cases)
                     or any_disp_limit):
        warn("the feasibility back-off controller (backoff_gain / "
             "damping_threshold) is configured but no load case sets a "
             "sigma_allow or d_allow limit -- it will never engage")

    # --- per-case weights & feasibility limits ---
    weights = [c.weight for c in cases]
    for c in cases:
        if c.weight < 0:
            err(f"load case {c.name!r}: weight must be >= 0: got {c.weight}")
    if weights and not any(w > 0 for w in weights):
        err("all load-case weights are zero -- the sensitivity has no objective")
    # sigma_allow / d_allow are optional: a blank limit (None) leaves that quantity
    # unconstrained. Only a set, non-positive limit is an error. Each displacement
    # constraint must name a node; a d_allow (when given) must be > 0.
    for c in cases:
        if c.sigma_allow is not None and c.sigma_allow <= 0:
            err(f"load case {c.name!r}: sigma_allow must be > 0: got {c.sigma_allow}")
        for j, dc in enumerate(c.disp_constraints):
            where = f"load case {c.name!r}: displacement constraint #{j + 1}"
            if dc.node_id is None:
                err(f"{where}: node id is required")
            if dc.d_allow is not None and dc.d_allow <= 0:
                err(f"{where}: d_allow must be > 0: got {dc.d_allow}")

    # --- growth regions (add-material boxes / spheres / cylinders) ---
    boxes = m.growth_boxes or []
    for i, b in enumerate(boxes):
        label = b.name or f"#{i + 1}"
        kind = b.shape_kind()
        if kind not in VALID_GROWTH_SHAPES:
            err(f"growth box {label!r}: unknown shape {b.shape!r} "
                f"(expected {', '.join(VALID_GROWTH_SHAPES)})")
            continue
        if kind == "box":
            # A deck_box_id box takes its corners from a /BOX/RECTA card at run
            # start, so its config coordinates aren't meaningful to check here.
            if b.deck_box_id is None:
                for axis in ("x", "y", "z"):
                    lo = getattr(b, f"{axis}_min")
                    hi = getattr(b, f"{axis}_max")
                    if lo > hi:
                        err(f"growth box {label!r}: {axis}_min ({lo}) > "
                            f"{axis}_max ({hi})")
                    elif lo == hi:
                        warn(f"growth box {label!r}: {axis}_min == {axis}_max "
                             f"({lo}) -- the region is degenerate and will select "
                             "no elements")
            _check_local_frame(b, label, err, warn)
        elif kind == "sphere":
            if b.radius <= 0:
                err(f"growth box {label!r}: sphere radius must be > 0: "
                    f"got {b.radius}")
        elif kind == "cylinder":
            if b.radius <= 0:
                err(f"growth box {label!r}: cylinder radius must be > 0: "
                    f"got {b.radius}")
            if (b.x1, b.y1, b.z1) == (b.x2, b.y2, b.z2):
                err(f"growth box {label!r}: cylinder axis end-points are "
                    "identical (zero-length axis)")
        elif kind == "polyhedron":
            _check_polyhedron_points(b, label, err)
        if kind != "polyhedron" and b.points:
            warn(f"growth box {label!r}: 'points' is only used by shape "
                 f"'polyhedron' -- ignored for shape {kind!r}")
    thr = m.growth_original_elem_max
    if thr is not None and (isinstance(thr, bool)
                            or not isinstance(thr, int) or thr <= 0):
        err("model.growth_original_elem_max must be a positive element id "
            f"(or blank): got {thr!r}")
    # Carve-off (the default) needs the original/expansion element-id boundary
    # to actually protect the part; without one it degrades to carving. One
    # config-level warning -- not per region, and not an error, so a plain
    # phase-1 config (regions over hand-pre-meshed expansion volume, no
    # boundary, no part overlap) keeps validating clean enough to run.
    if thr is None and any(not b.carve for b in boxes):
        warn("growth regions with carve off (the default) but no "
             "model.growth_original_elem_max -- no original/expansion "
             "element-id boundary is known, so every in-region element starts "
             "void and a region overlapping the part will carve it (as if "
             "carve were true). The growth-mesh step records the boundary "
             "when pointing the config at the extended decks; set it manually "
             "for a hand-pre-meshed deck, or set carve: true to make carving "
             "explicit")
    if boxes and cfg.optimizer_name() == "beso" \
            and cfg.beso.max_add_ratio < cfg.beso.evolution_rate:
        warn(f"growth boxes with beso.max_add_ratio={cfg.beso.max_add_ratio} < "
             f"evolution_rate={cfg.beso.evolution_rate}: per-iteration growth "
             "into the boxes is capped below the feasibility back-off step; "
             "consider max_add_ratio >= evolution_rate")

    # --- growth keep-out deck (forbidden growth space: nearby parts) ---
    if m.growth_keepout_rad:
        clr = m.growth_keepout_clearance_mm
        if isinstance(clr, bool) or not isinstance(clr, (int, float)) or clr < 0:
            err("model.growth_keepout_clearance_mm must be >= 0: "
                f"got {clr!r}")
        if not boxes:
            warn("model.growth_keepout_rad is set but no growth regions are "
                 "configured -- the keep-out has no candidates to remove (no-op)")
        ko_path = Path(m.growth_keepout_rad)
        if not ko_path.is_absolute():
            ko_path = case_dir / m.growth_keepout_rad
        if not ko_path.exists():
            err(f"growth keep-out deck not found: {ko_path}")
        else:
            try:
                _tets, _nodes, found = read_solid_geometry(
                    ko_path, m.growth_keepout_part_ids or None)
            except Exception as exc:  # noqa: BLE001
                err(f"growth keep-out deck could not be parsed ({ko_path}): {exc}")
            else:
                want = [int(p) for p in (m.growth_keepout_part_ids or [])]
                missing = [p for p in want if p not in found]
                if missing:
                    warn("growth keep-out deck has no solid elements for part "
                         f"id(s) {missing} (found {found}); those parts "
                         "contribute nothing to the keep-out")

    return problems


def validate_config(cfg: Config, *, raw: dict | None = None,
                    probe_docker_image: bool = False) -> list[str]:
    """Human-readable, severity-prefixed validation problems for *cfg*.

    An empty list means the config is clean. Each string reads ``"error: ..."``
    or ``"warning: ..."``; use :func:`check_config` when you need the structured
    severities (e.g. to decide whether to block a launch).
    """
    return [str(p) for p in check_config(
        cfg, raw=raw, probe_docker_image=probe_docker_image)]

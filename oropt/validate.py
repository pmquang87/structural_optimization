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
from typing import Callable, Iterable

from .config import Config, unknown_keys
from .deck import read_solid_geometry
from .runner import backend_problems

ERROR = "error"
WARNING = "warning"

VALID_OPTIMIZERS = ("beso", "levelset", "tobs", "hca", "saip")
VALID_GROWTH_SHAPES = ("box", "cylinder", "sphere", "polyhedron", "deck")
VALID_SENSITIVITIES = ("energy", "vonmises", "blend", "tdsa")


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


def _check_deck_region(b, label, case_dir, err, warn) -> None:
    """Validate a ``shape="deck"`` region: a ``region_rad`` deck that exists
    (relative to ``case_dir``), parses to solid geometry, actually contains the
    selected part ids, and a non-negative clearance. Mirrors the keep-out deck
    checks (:func:`oropt.deck.read_solid_geometry`) — the two are geometric mirrors
    (a positive region vs a forbidden one)."""
    rad = (b.region_rad or "").strip()
    if not rad:
        err(f"growth box {label!r}: shape 'deck' needs region_rad -- a Radioss "
            "deck whose parts' geometry defines the growth region")
        return
    clr = b.region_clearance_mm
    if isinstance(clr, bool) or not isinstance(clr, (int, float)) or clr < 0:
        err(f"growth box {label!r}: region_clearance_mm must be >= 0: got {clr!r}")
    path = Path(rad)
    if not path.is_absolute():
        path = case_dir / rad
    if not path.exists():
        err(f"growth box {label!r}: region deck not found: {path}")
        return
    try:
        _tets, _nodes, _surf, found = read_solid_geometry(
            path, b.region_part_ids or None)
    except Exception as exc:  # noqa: BLE001
        err(f"growth box {label!r}: region deck could not be parsed ({path}): {exc}")
        return
    want = [int(p) for p in (b.region_part_ids or [])]
    missing = [p for p in want if p not in found]
    if missing:
        warn(f"growth box {label!r}: region deck has no solid elements for part "
             f"id(s) {missing} (found {found}); those parts contribute nothing "
             "to the region")


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


_Say = Callable[[str], None]


def _collector() -> tuple[list[Problem], _Say, _Say]:
    """A fresh problem list plus its ``err``/``warn`` appenders (one per helper)."""
    problems: list[Problem] = []

    def err(msg: str) -> None:
        problems.append(Problem(ERROR, msg))

    def warn(msg: str) -> None:
        problems.append(Problem(WARNING, msg))

    return problems, err, warn


def _check_unknown_keys(raw: dict | None) -> list[Problem]:
    """Unrecognised config keys (silently dropped by ``Config.from_dict``).

    An error, not a warning: a typo'd knob reverts to its default and the run
    would silently optimise the wrong thing for hours. Blocking the launch is
    cheaper than discovering it afterwards. No-op when *raw* is ``None``.
    """
    problems, err, _warn = _collector()
    if raw is not None:
        for key in unknown_keys(raw):
            err(f"unrecognised config key {key!r} -- typo or wrong section? "
                "(it would be ignored and its default used). Remove or correct it.")
    return problems


def _check_optimizer(cfg: Config) -> list[Problem]:
    """The optimiser selector: must name one of the known algorithms."""
    problems, err, _warn = _collector()
    if cfg.optimizer_name() not in VALID_OPTIMIZERS:
        err(f"optimizer must be one of {', '.join(VALID_OPTIMIZERS)} "
            f"(got {cfg.optimizer!r})")
    return problems


def _check_model_and_decks(cfg: Config) -> list[Problem]:
    """Model directory and per-load-case decks: the case dir must be a real
    directory, every case needs a stem with existing starter/engine decks, and
    stems must be distinct (the whole file layout is keyed by them)."""
    problems, err, _warn = _collector()
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
    # The whole multi-case file layout is keyed by stem (per-case solve dirs,
    # iter_NNNN/<stem>/ archives, iter-0 reuse, d3plot names): two cases sharing
    # one would silently overwrite each other's artefacts every iteration.
    stems = [c.stem.strip() for c in cases if (c.stem or "").strip()]
    dup_stems = sorted({s for s in stems if stems.count(s) > 1})
    if dup_stems:
        err("load cases must have distinct deck stems -- duplicated: "
            + ", ".join(repr(s) for s in dup_stems)
            + " (their per-iteration archives / iter-0 reuse / d3plots would "
              "overwrite each other)")
    return problems


def _check_run_folder(cfg: Config) -> list[Problem]:
    """The run/output folder must be creatable (``mkdir -p``-able)."""
    problems, err, _warn = _collector()
    run_folder = Path(cfg.run_folder())
    if not _creatable(run_folder):
        err(f"run folder is not creatable: {run_folder.resolve()}")
    return problems


def _check_backend(cfg: Config, probe_docker_image: bool) -> list[Problem]:
    """Solver backend readiness: missing executables/CLI (shared source of truth
    with ``runner.run_solver`` via :func:`backend_problems`), the optional Docker
    image probe, and the native backend's np=1 requirement."""
    problems, err, warn = _collector()
    for msg in backend_problems(cfg):
        err(msg)
    if cfg.docker.enabled:
        if probe_docker_image and _docker_image_present(cfg) is False:
            warn(f"docker image not found locally: {cfg.docker.image} "
                 "(it will be pulled on first run, or the run will fail)")
    elif cfg.run.np != 1:
        warn(f"run.np={cfg.run.np}: the native implicit + solid-contact solver "
             "requires np=1 (it segfaults otherwise)")
    return problems


def _check_run_limits(cfg: Config) -> list[Problem]:
    """Run-level watchdog/budget knobs: the engine soft-vs-hard timeout ordering,
    the wall-clock budget, and the divergence-abort counters."""
    problems, err, warn = _collector()
    if cfg.run.engine_soft_timeout_s > 0 \
            and cfg.run.engine_soft_timeout_s >= cfg.run.engine_timeout_s:
        warn(f"run.engine_soft_timeout_s={cfg.run.engine_soft_timeout_s:g} >= "
             f"engine_timeout_s={cfg.run.engine_timeout_s:g}: the hard kill "
             "fires first (and fails the run), so the soft budget never "
             "triggers the treat-as-infeasible back-off")
    if cfg.run.max_wall_hours < 0:
        err(f"run.max_wall_hours must be >= 0 (0 = unlimited): got "
            f"{cfg.run.max_wall_hours}")
    if cfg.run.diverge_fail_after < 1:
        err(f"run.diverge_fail_after must be >= 1: got "
            f"{cfg.run.diverge_fail_after}")
    if cfg.run.diverge_max_cycles < 0:
        err(f"run.diverge_max_cycles must be >= 0 (0 = off): got "
            f"{cfg.run.diverge_max_cycles}")
    return problems


def _check_optimizer_knobs(cfg: Config) -> list[Problem]:
    """Numeric sanity of the *active* optimiser block's knobs -- volume target,
    rates, the feasibility back-off controller (gain/cap/floor, gate vs
    multipoint), and the optimiser-specific extras (level-set, HCA, TOBS, SAIP)
    -- plus the warnings for controllers configured without any feasibility
    limit to react to."""
    problems, err, warn = _collector()
    cases = cfg.load_case_list()
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
    if getattr(opt, "sensitivity", "energy") not in VALID_SENSITIVITIES:
        # map_sensitivity silently falls back to "energy" for an unknown value --
        # a typo'd mode would run the whole optimisation on the wrong ranking,
        # so block it here instead.
        err(f"sensitivity must be one of {', '.join(VALID_SENSITIVITIES)}: "
            f"got {opt.sensitivity!r}")
    if opt.backoff_mode not in ("gate", "multipoint"):
        err(f"backoff_mode must be 'gate' or 'multipoint': "
            f"got {opt.backoff_mode!r}")
    if opt.multipoint_window < 2:
        err(f"multipoint_window must be >= 2 (a 1-point fit has no slope): "
            f"got {opt.multipoint_window}")
    if not (0 < opt.utilization_target <= 1):
        err(f"utilization_target must be in (0, 1] (1.0 = right at the "
            f"limits): got {opt.utilization_target}")
    # optimiser-specific knobs (guarded getattr: only the owning block has them)
    ur = getattr(opt, "update_rule", "advect")
    if ur not in ("advect", "rde"):
        err(f"levelset.update_rule must be 'advect' or 'rde': got {ur!r}")
    if getattr(opt, "diffusion", 0.0) < 0:
        err(f"levelset.diffusion must be >= 0: got {opt.diffusion}")
    if getattr(opt, "radius_start", 0.0) < 0:
        err(f"hca.radius_start must be >= 0 (0 = fixed filter_radius, "
            f"classic HCA): got {opt.radius_start}")
    elif 0.0 < getattr(opt, "radius_start", 0.0) <= opt.filter_radius:
        warn(f"hca.radius_start={opt.radius_start} is not above "
             f"filter_radius={opt.filter_radius} -- the MHCA neighbourhood "
             "schedule only decays, so it is a no-op (set radius_start > "
             "filter_radius, or 0 to silence this)")
    if getattr(opt, "radius_iters", 1) < 1:
        err(f"hca.radius_iters must be >= 1: got {opt.radius_iters}")
    if getattr(opt, "radius_steps", 2) < 2:
        err(f"hca.radius_steps must be >= 2: got {opt.radius_steps}")
    fl = getattr(opt, "flip_limit", None)
    if fl is not None and not (0 < fl <= 1):
        err(f"flip_limit must be in (0, 1]: got {fl}")
    od = getattr(opt, "oscillation_damping", None)
    if od is not None and not (0 < od <= 1):
        err(f"saip.oscillation_damping must be in (0, 1] (1.0 = off): "
            f"got {od}")
    any_disp_limit = any(dc.d_allow is not None
                         for c in cases for dc in c.disp_constraints)
    if (opt.backoff_gain > 0 or opt.damping_threshold < 1) \
            and not (any(c.sigma_allow is not None for c in cases)
                     or any_disp_limit):
        warn("the feasibility back-off controller (backoff_gain / "
             "damping_threshold) is configured but no load case sets a "
             "sigma_allow or d_allow limit -- it will never engage")
    if opt.backoff_mode == "multipoint" \
            and not (any(c.sigma_allow is not None for c in cases)
                     or any_disp_limit):
        warn("backoff_mode is 'multipoint' but no load case sets a "
             "sigma_allow or d_allow limit -- there is no violation signal "
             "to fit, so the controller will always fall back to the gate")
    return problems


def _check_load_cases(cfg: Config) -> list[Problem]:
    """Per-load-case weights and feasibility limits: non-negative weights with at
    least one positive, optional sigma_allow > 0, and displacement constraints
    each naming a node with an optional d_allow > 0."""
    problems, err, _warn = _collector()
    cases = cfg.load_case_list()
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
    return problems


def _growth_boxes(cfg: Config) -> list:
    """The growth regions this run will actually use: the configured boxes, or
    none at all when growth is switched off (the boxes are retained in the config
    so the GUI can toggle back on, but ignored at run time)."""
    m = cfg.model
    return (m.growth_boxes or []) if getattr(m, "growth_enabled", True) else []


def _check_growth(cfg: Config) -> list[Problem]:
    """Growth (add-material) regions: per-shape geometry sanity (box corners and
    local frame, sphere/cylinder radii and axes, polyhedron point sets), the
    original/expansion element-id boundary, and the carve / BESO add-ratio
    interplay warnings. Skipped entirely when growth is switched off."""
    problems, err, warn = _collector()
    m = cfg.model
    boxes = _growth_boxes(cfg)
    case_dir = Path(m.case_dir).resolve()   # deck regions resolve relative to it
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
        elif kind == "deck":
            _check_deck_region(b, label, case_dir, err, warn)
        if kind != "polyhedron" and b.points:
            warn(f"growth box {label!r}: 'points' is only used by shape "
                 f"'polyhedron' -- ignored for shape {kind!r}")
        if getattr(b, "forbid", False) and b.carve:
            warn(f"growth box {label!r}: carve is ignored for a negative "
                 "(forbid=true) region -- a negative region only forbids "
                 "growth, it never carves the part")
    pos_boxes = [b for b in boxes if not getattr(b, "forbid", False)]
    neg_boxes = [b for b in boxes if getattr(b, "forbid", False)]
    if neg_boxes and not pos_boxes:
        warn("every growth region is negative (forbid=true) -- there are no "
             "positive add-material regions, so growth adds nothing and the "
             "negative regions have no candidates to forbid (no-op)")
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
    if thr is None and any(not b.carve for b in pos_boxes):
        warn("growth regions with carve off (the default) but no "
             "model.growth_original_elem_max -- no original/expansion "
             "element-id boundary is known, so every in-region element starts "
             "void and a region overlapping the part will carve it (as if "
             "carve were true). The growth-mesh step records the boundary "
             "when pointing the config at the extended decks; set it manually "
             "for a hand-pre-meshed deck, or set carve: true to make carving "
             "explicit")
    if pos_boxes and cfg.optimizer_name() == "beso" \
            and cfg.beso.max_add_ratio < cfg.beso.evolution_rate:
        warn(f"growth boxes with beso.max_add_ratio={cfg.beso.max_add_ratio} < "
             f"evolution_rate={cfg.beso.evolution_rate}: per-iteration growth "
             "into the boxes is capped below the feasibility back-off step; "
             "consider max_add_ratio >= evolution_rate")
    return problems


def _check_growth_keepout(cfg: Config) -> list[Problem]:
    """The optional growth keep-out deck (forbidden growth space: nearby parts):
    clearance-band sanity, the deck must exist and parse, and its requested part
    ids must actually contribute solid elements."""
    problems, err, warn = _collector()
    m = cfg.model
    boxes = _growth_boxes(cfg)
    case_dir = Path(m.case_dir).resolve()
    if m.growth_keepout_rad:
        clr = m.growth_keepout_clearance_mm
        if isinstance(clr, bool) or not isinstance(clr, (int, float)) \
                or not math.isfinite(clr):
            # NaN passed the old `< 0` check (every comparison with NaN is
            # False) and then silently disabled the clearance band downstream.
            err("model.growth_keepout_clearance_mm must be a finite number "
                f"(negative = allowed penetration depth): got {clr!r}")
        elif clr < 0:
            warn(f"growth_keepout_clearance_mm={clr:g}: NEGATIVE clearance -- "
                 f"growth may deliberately penetrate up to {-clr:g} into the "
                 "neighbour parts (an interference/overlap band). If a gap was "
                 "intended, use a positive value")
        if not any(not getattr(b, "forbid", False) for b in boxes):
            warn("model.growth_keepout_rad is set but no positive growth regions "
                 "are configured -- the keep-out has no candidates to remove "
                 "(no-op)")
        ko_path = Path(m.growth_keepout_rad)
        if not ko_path.is_absolute():
            ko_path = case_dir / m.growth_keepout_rad
        if not ko_path.exists():
            err(f"growth keep-out deck not found: {ko_path}")
        else:
            try:
                _tets, _nodes, _surf, found = read_solid_geometry(
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


def check_config(cfg: Config, *, raw: dict | None = None,
                 probe_docker_image: bool = False) -> list[Problem]:
    """Structured validation of *cfg* -- the engine behind :func:`validate_config`.

    Pure and fast (no solver, no file writes). The only optional side effect is a
    short ``docker image inspect`` when *probe_docker_image* is set and the Docker
    backend is selected; it is off by default so the check stays hermetic.

    Pass *raw* (the mapping the config was parsed from, e.g.
    :meth:`Config.read_yaml_dict`) to also **error** on unrecognised keys that
    :meth:`Config.from_dict` silently dropped -- a typo'd or misplaced knob (e.g.
    ``evolution_ratte``) would otherwise revert to its default unnoticed and the
    run would spend hours optimising the wrong thing, so a stray key blocks the
    launch rather than merely warning.

    Composes the ``_check_*`` helpers; their order (and each helper's internal
    order) is the report order, so it is part of the observable behaviour.
    """
    return [
        *_check_unknown_keys(raw),
        *_check_optimizer(cfg),
        *_check_model_and_decks(cfg),
        *_check_run_folder(cfg),
        *_check_backend(cfg, probe_docker_image),
        *_check_run_limits(cfg),
        *_check_optimizer_knobs(cfg),
        *_check_load_cases(cfg),
        *_check_growth(cfg),
        *_check_growth_keepout(cfg),
    ]


def validate_config(cfg: Config, *, raw: dict | None = None,
                    probe_docker_image: bool = False) -> list[str]:
    """Human-readable, severity-prefixed validation problems for *cfg*.

    An empty list means the config is clean. Each string reads ``"error: ..."``
    or ``"warning: ..."``; use :func:`check_config` when you need the structured
    severities (e.g. to decide whether to block a launch).
    """
    return [str(p) for p in check_config(
        cfg, raw=raw, probe_docker_image=probe_docker_image)]

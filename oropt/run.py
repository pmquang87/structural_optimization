"""Command-line entry point: ``python -m oropt.run --config <cfg.yaml>``.

Runs the BESO loop headless. The Streamlit GUI launches this same command as a
subprocess and monitors the status files it writes.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import sys
from pathlib import Path
from typing import Callable, Iterator

from .config import Config
from .loop import run_optimization
from .status import RUN_LOG
from .validate import check_config, has_errors


@contextlib.contextmanager
def _tee_log(work: Path, resume: bool) -> Iterator[Callable[[str], None]]:
    """Yield a ``log(str)`` that prints to stdout *and* appends to ``<work>/run.log``.

    The GUI launches this module detached with ``stdout``/``stderr`` -> DEVNULL,
    so without an on-disk copy the loop's progress and — critically — every
    best-effort post-run step (d3plot / smooth / animate / report, each of which
    only *logs* its skip reason) vanish, leaving a silent failure impossible to
    diagnose after the fact. A resumed run appends; a fresh run truncates so the
    log matches the run it describes. Best-effort: if the file can't be opened we
    degrade to stdout-only rather than fail the run.
    """
    fh = None
    try:
        fh = open(work / RUN_LOG, "a" if resume else "w",
                  encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[oropt] could not open {RUN_LOG}: {exc} - logging to stdout only",
              flush=True)

    def log(s: str) -> None:
        print(s, flush=True)
        if fh is not None:
            ts = _dt.datetime.now().isoformat(timespec="seconds")
            fh.write(f"{ts} {s}\n")
            fh.flush()

    try:
        yield log
    finally:
        if fh is not None:
            fh.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="oropt", description="OpenRadioss-coupled BESO topology optimisation")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--resume", action="store_true",
                    help="resume from checkpoint.npz in work_dir")
    ap.add_argument("--work-dir", default=None,
                    help="override the config's run/output folder for this run "
                         "(the queue uses this to give colliding runs their own "
                         "folder without editing the shared config)")
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip the fail-fast config check (launch even on errors)")
    args = ap.parse_args(argv)

    raw = Config.read_yaml_dict(args.config)
    cfg = Config.from_dict(raw)
    if args.work_dir:                       # CLI override wins over the config's
        cfg.work_dir = args.work_dir        # work_dir (and its blank->case_dir default)

    # Fail fast: a bad config is caught here in ~1 s, not after a 13-min solve or
    # hours into the loop. Errors abort before launch; warnings (incl. unrecognised
    # keys from `raw`) are printed only.
    if not args.skip_validate:
        problems = check_config(cfg, raw=raw, probe_docker_image=True)
        for p in problems:
            print(f"[oropt] {p}", flush=True)
        if has_errors(problems):
            print("[oropt] config has errors -- aborting before launch "
                  "(use --skip-validate to override)", flush=True)
            return 2

    # Tee the run's log to <work>/run.log so nothing (least of all a best-effort
    # post-run step's skip reason) is lost to the GUI's DEVNULL launch. cfg.work()
    # creates the folder if needed.
    with _tee_log(cfg.work(), args.resume) as log:
        status = run_optimization(cfg, resume=args.resume, log=log)
        log(f"[oropt] finished: state={status.state} iter={status.iteration} "
            f"vf={status.volume_fraction:.3f} msg={status.message!r}")
    return 0 if status.state in ("converged", "stopped") else 1


if __name__ == "__main__":
    sys.exit(main())

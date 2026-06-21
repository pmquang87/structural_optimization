"""Command-line entry point: ``python -m oropt.run --config <cfg.yaml>``.

Runs the BESO loop headless. The Streamlit GUI launches this same command as a
subprocess and monitors the status files it writes.
"""
from __future__ import annotations

import argparse
import sys

from .config import Config
from .loop import run_optimization
from .validate import check_config, has_errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="oropt", description="OpenRadioss-coupled BESO topology optimisation")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--resume", action="store_true",
                    help="resume from checkpoint.npz in work_dir")
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip the fail-fast config check (launch even on errors)")
    args = ap.parse_args(argv)

    raw = Config.read_yaml_dict(args.config)
    cfg = Config.from_dict(raw)

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

    status = run_optimization(cfg, resume=args.resume,
                              log=lambda s: print(s, flush=True))
    print(f"[oropt] finished: state={status.state} iter={status.iteration} "
          f"vf={status.volume_fraction:.3f} msg={status.message!r}")
    return 0 if status.state in ("converged", "stopped") else 1


if __name__ == "__main__":
    sys.exit(main())

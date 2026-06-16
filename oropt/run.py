"""Command-line entry point: ``python -m oropt.run --config <cfg.yaml>``.

Runs the BESO loop headless. The Streamlit GUI launches this same command as a
subprocess and monitors the status files it writes.
"""
from __future__ import annotations

import argparse
import sys

from .config import Config
from .loop import run_optimization


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="oropt", description="OpenRadioss-coupled BESO topology optimisation")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--resume", action="store_true",
                    help="resume from checkpoint.npz in work_dir")
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    status = run_optimization(cfg, resume=args.resume,
                              log=lambda s: print(s, flush=True))
    print(f"[oropt] finished: state={status.state} iter={status.iteration} "
          f"vf={status.volume_fraction:.3f} msg={status.message!r}")
    return 0 if status.state in ("converged", "stopped") else 1


if __name__ == "__main__":
    sys.exit(main())

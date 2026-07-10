# Cantilever demo case

A tiny TET4 cantilever beam (100x25x25 mm, 720 nodes, 2850 tets, design part `60000000`) with its x=0 face clamped via `/GRNOD/NODE/60000000` and a tip-load node group `/GRNOD/NODE/60000001` (suggested displacement-constraint node: `60000699`, the tip-face centre).
Generated deterministically by `python scripts/make_demo_case.py` (rerun it to regenerate; options: `--nx/--ny/--nz/--lx/--ly/--lz`).
It pairs with the synthetic demo backend (`oropt/demo.py`), so the whole loop/monitor/report/smoothing/GIF pipeline runs on this deck with zero OpenRadioss installed.

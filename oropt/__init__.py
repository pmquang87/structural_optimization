"""oropt — OpenRadioss-coupled topology optimization via BESO.

Bi-directional Evolutionary Structural Optimization that drives the *real*
OpenRadioss implicit nonlinear model: each iteration solves the deck, ranks
elements by internal-energy density read from ``/ANIM/ELEM/ENER``, deletes the
least-important ones (with add-back), and re-runs — removing material while the
high-fidelity solver still reports peak von-Mises and loaded-node displacement
within their limits.

See ``README.md`` for the architecture and honest caveats.
"""

__version__ = "0.1.0"

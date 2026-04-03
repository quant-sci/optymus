"""Vehicle Routing Problem (VRP) module.

Provides solvers for the Capacitated Vehicle Routing Problem (CVRP)
using constructive heuristics and local search improvement.
"""

from optymus.routing._utils import compute_distance_matrix
from optymus.routing._vrp import VRPSolver, solve_vrp

__all__ = [
    "VRPSolver",
    "solve_vrp",
    "compute_distance_matrix",
]

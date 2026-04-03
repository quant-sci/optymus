Vehicle Routing Problem (VRP)
=============================

This module provides solvers for the **Capacitated Vehicle Routing Problem (CVRP)**
using constructive heuristics and local search improvement.

Given a depot, a set of customers with demands, and a fleet of capacity-constrained
vehicles, the solver finds routes that minimize total travel distance while serving
every customer exactly once.

Core Solver
-----------

.. autosummary::
    :toctree: .generated/

    optymus.routing.VRPSolver
    optymus.routing.solve_vrp

Utilities
---------

.. autosummary::
    :toctree: .generated/

    optymus.routing.compute_distance_matrix

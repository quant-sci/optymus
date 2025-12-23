"""
Finite Element Method (FEM) Module for Topology Optimization.

This module provides tools for structural analysis and topology optimization
using the Finite Element Method with polygonal meshes.

Main Components
---------------
ComplianceObjective
    SIMP-based compliance objective function for topology optimization

topology_optimization
    High-level function to run topology optimization with optymus methods

optimality_criteria
    Specialized OC method for compliance minimization

solve_fem
    Low-level FEM solver for structural analysis

Examples
--------
>>> from optymus.benchmark import MbbDomain
>>> from optymus.methods import polymesher
>>> from optymus.fem import topology_optimization, plot_topology
>>>
>>> # Generate mesh
>>> mesh = polymesher(domain=MbbDomain, n_elements=200, max_iter=100, plot=False)
>>>
>>> # Run topology optimization
>>> results = topology_optimization(
...     mesh=mesh,
...     method="adam",
...     volume_fraction=0.5,
...     max_iter=100
... )
>>>
>>> # Visualize result
>>> plot_topology(mesh, results["densities"])
"""

from ._compliance import ComplianceObjective, create_compliance_objective
from ._topology_opt import (
    topology_optimization,
    optimality_criteria,
    plot_topology,
    plot_convergence,
)
from ._polygonal_fem import (
    solve_fem,
    plane_stress_D,
    polygon_area,
    polygon_centroid,
    assemble_global_stiffness,
    apply_boundary_conditions,
    apply_loads,
    compute_element_compliance,
)

__all__ = [
    # High-level API
    "topology_optimization",
    "optimality_criteria",
    "ComplianceObjective",
    "create_compliance_objective",
    # Visualization
    "plot_topology",
    "plot_convergence",
    # Low-level FEM (2D polygonal)
    "solve_fem",
    "plane_stress_D",
    "polygon_area",
    "polygon_centroid",
    "assemble_global_stiffness",
    "apply_boundary_conditions",
    "apply_loads",
    "compute_element_compliance",
]

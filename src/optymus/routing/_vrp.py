import time

import numpy as np

from optymus.methods.utils._result import OptimizeResult

from ._local_search import improve_solution
from ._savings import clarke_wright_savings
from ._utils import (
    compute_distance_matrix,
    route_distance,
    route_load,
    total_distance,
    validate_vrp_inputs,
)


class VRPSolver:
    """Capacitated Vehicle Routing Problem solver.

    Parameters
    ----------
    distance_matrix : array-like, shape (n, n), optional
        Pairwise distance/cost matrix. Provide this OR ``coordinates``.
    coordinates : array-like, shape (n, 2), optional
        Node coordinates (Euclidean distances computed automatically).
    demands : array-like, shape (n,)
        Demand at each node.  ``demands[depot]`` should be 0.
    vehicle_capacity : float
        Maximum load per vehicle.
    num_vehicles : int, optional
        Number of vehicles.  If *None*, uses the minimum feasible.
    depot : int
        Index of the depot node (default 0).
    verbose : bool
        Show progress information (default True).
    """

    def __init__(
        self,
        distance_matrix=None,
        coordinates=None,
        demands=None,
        vehicle_capacity=None,
        num_vehicles=None,
        depot=0,
        verbose=True,
    ):
        if distance_matrix is None and coordinates is None:
            raise ValueError("Provide either distance_matrix or coordinates")

        if distance_matrix is None:
            distance_matrix = compute_distance_matrix(coordinates)

        if demands is None:
            raise ValueError("demands is required")
        if vehicle_capacity is None:
            raise ValueError("vehicle_capacity is required")

        self.distance_matrix, self.demands, self.num_vehicles = validate_vrp_inputs(
            distance_matrix, demands, vehicle_capacity, num_vehicles, depot,
        )
        self.vehicle_capacity = vehicle_capacity
        self.depot = depot
        self.verbose = verbose
        self.coordinates = np.asarray(coordinates) if coordinates is not None else None

    def solve(self, method="savings", max_iter=100):
        """Solve the CVRP.

        Parameters
        ----------
        method : str
            Algorithm to use.  Currently ``"savings"`` (Clarke-Wright + 2-opt).
        max_iter : int
            Maximum local-search improvement rounds.

        Returns
        -------
        OptimizeResult
        """
        if method != "savings":
            raise ValueError(f"Unknown method '{method}'. Available: 'savings'")

        start_time = time.time()

        # Phase 1: construct initial solution
        routes = clarke_wright_savings(
            self.distance_matrix,
            self.demands,
            self.vehicle_capacity,
            self.num_vehicles,
            self.depot,
        )
        initial_cost = total_distance(routes, self.distance_matrix, self.depot)

        if self.verbose:
            print(f"Initial solution: {len(routes)} routes, distance = {initial_cost:.4f}")

        # Phase 2: improve with local search
        routes, num_iter = improve_solution(
            routes,
            self.distance_matrix,
            self.demands,
            self.vehicle_capacity,
            self.depot,
            max_iter,
        )
        final_cost = total_distance(routes, self.distance_matrix, self.depot)

        elapsed = time.time() - start_time

        if self.verbose:
            improvement = (initial_cost - final_cost) / initial_cost * 100 if initial_cost > 0 else 0
            print(f"Final solution:   {len(routes)} routes, distance = {final_cost:.4f} "
                  f"({improvement:.1f}% improvement)")

        # Build per-route details
        route_details = []
        for r in routes:
            route_details.append({
                "customers": list(r),
                "distance": route_distance(r, self.distance_matrix, self.depot),
                "load": route_load(r, self.demands),
            })

        return OptimizeResult({
            "method_name": "CVRP (Clarke-Wright Savings + 2-opt)",
            "routes": [list(r) for r in routes],
            "route_details": route_details,
            "num_vehicles_used": len(routes),
            "total_distance": final_cost,
            "initial_distance": initial_cost,
            "improvement": (initial_cost - final_cost) / initial_cost if initial_cost > 0 else 0.0,
            "fmin": final_cost,
            "xopt": [list(r) for r in routes],
            "num_iter": num_iter,
            "termination_reason": "local_search_converged",
            "time": elapsed,
        })


def solve_vrp(
    distance_matrix=None,
    coordinates=None,
    demands=None,
    vehicle_capacity=None,
    num_vehicles=None,
    depot=0,
    method="savings",
    max_iter=100,
    verbose=True,
):
    """Solve a Capacitated Vehicle Routing Problem.

    Convenience function that creates a :class:`VRPSolver` and calls
    :meth:`~VRPSolver.solve`.

    Parameters
    ----------
    distance_matrix : array-like, shape (n, n), optional
        Pairwise distance/cost matrix.  Provide this OR ``coordinates``.
    coordinates : array-like, shape (n, 2), optional
        Node coordinates (Euclidean distances computed automatically).
    demands : array-like, shape (n,)
        Demand at each node.  ``demands[depot]`` should be 0.
    vehicle_capacity : float
        Maximum load per vehicle.
    num_vehicles : int, optional
        Number of vehicles.  If *None*, uses the minimum feasible.
    depot : int
        Index of the depot node (default 0).
    method : str
        Algorithm: ``"savings"`` (Clarke-Wright + 2-opt).
    max_iter : int
        Maximum local-search improvement rounds.
    verbose : bool
        Print progress information.

    Returns
    -------
    OptimizeResult
    """
    solver = VRPSolver(
        distance_matrix=distance_matrix,
        coordinates=coordinates,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        num_vehicles=num_vehicles,
        depot=depot,
        verbose=verbose,
    )
    return solver.solve(method=method, max_iter=max_iter)

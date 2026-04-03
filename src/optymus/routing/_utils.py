import numpy as np


def compute_distance_matrix(coordinates):
    """Compute Euclidean distance matrix from coordinate array.

    Parameters
    ----------
    coordinates : array-like, shape (n, 2) or (n, 3)
        Node coordinates.

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric distance matrix.
    """
    coords = np.asarray(coordinates, dtype=float)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def validate_vrp_inputs(distance_matrix, demands, vehicle_capacity, num_vehicles, depot):
    """Validate and normalize CVRP inputs.

    Returns
    -------
    distance_matrix, demands, num_vehicles : validated arrays and int
    """
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distance_matrix must be square, got shape {D.shape}")

    demands = np.asarray(demands, dtype=float)
    if demands.shape != (n,):
        raise ValueError(
            f"demands must have length {n} (matching distance_matrix), got {demands.shape}"
        )

    if vehicle_capacity <= 0:
        raise ValueError("vehicle_capacity must be positive")

    if np.any(demands[np.arange(n) != depot] > vehicle_capacity):
        raise ValueError(
            "At least one customer demand exceeds vehicle_capacity"
        )

    total_demand = np.sum(demands) - demands[depot]

    if num_vehicles is None:
        num_vehicles = int(np.ceil(total_demand / vehicle_capacity))

    if num_vehicles * vehicle_capacity < total_demand:
        raise ValueError(
            f"Infeasible: {num_vehicles} vehicles x {vehicle_capacity} capacity "
            f"= {num_vehicles * vehicle_capacity} < total demand {total_demand}"
        )

    return D, demands, num_vehicles


def route_distance(route, distance_matrix, depot=0):
    """Total distance for a single route: depot -> customers -> depot.

    Parameters
    ----------
    route : list[int]
        Customer indices (excluding depot).
    distance_matrix : np.ndarray
    depot : int

    Returns
    -------
    float
    """
    if not route:
        return 0.0
    d = distance_matrix[depot, route[0]]
    for i in range(len(route) - 1):
        d += distance_matrix[route[i], route[i + 1]]
    d += distance_matrix[route[-1], depot]
    return float(d)


def route_load(route, demands):
    """Total demand served by a route."""
    return float(sum(demands[c] for c in route))


def total_distance(routes, distance_matrix, depot=0):
    """Sum of distances across all routes."""
    return sum(route_distance(r, distance_matrix, depot) for r in routes)

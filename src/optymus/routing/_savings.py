import numpy as np


def clarke_wright_savings(distance_matrix, demands, vehicle_capacity, num_vehicles, depot=0):
    """Construct a CVRP solution using the Clarke-Wright parallel savings algorithm.

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (n, n)
    demands : np.ndarray, shape (n,)
    vehicle_capacity : float
    num_vehicles : int
        Maximum number of vehicles.
    depot : int

    Returns
    -------
    list[list[int]]
        Routes as lists of customer indices (excluding depot).
    """
    n = distance_matrix.shape[0]
    customers = [i for i in range(n) if i != depot]

    if not customers:
        return []

    # Compute savings: s(i,j) = d(depot,i) + d(depot,j) - d(i,j)
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = (distance_matrix[depot, i]
                     + distance_matrix[depot, j]
                     - distance_matrix[i, j])
                savings.append((s, i, j))

    # Sort by savings descending
    savings.sort(key=lambda x: -x[0])

    # Initialize: one route per customer
    route_of = {}       # customer -> route index
    routes = {}         # route index -> list of customers
    route_load = {}     # route index -> total demand
    route_ends = {}     # route index -> (first_customer, last_customer)

    for idx, c in enumerate(customers):
        route_of[c] = idx
        routes[idx] = [c]
        route_load[idx] = demands[c]
        route_ends[idx] = (c, c)

    def _can_merge(ri, rj, i, j):
        """Check if routes ri and rj can be merged by linking i-j."""
        if ri == rj:
            return False
        # i must be at an endpoint of ri, j must be at an endpoint of rj
        ei = route_ends[ri]
        ej = route_ends[rj]
        if i not in ei or j not in ej:
            return False
        # Capacity check
        if route_load[ri] + route_load[rj] > vehicle_capacity:
            return False
        return True

    def _merge(ri, rj, i, j):
        """Merge route rj into ri by linking customer i to customer j."""
        ei_first, ei_last = route_ends[ri]
        ej_first, ej_last = route_ends[rj]

        r_i = routes[ri]
        r_j = routes[rj]

        # Orient the routes so the link i-j is at the junction
        if i == ei_last and j == ej_first:
            merged = r_i + r_j
        elif i == ei_last and j == ej_last:
            merged = r_i + r_j[::-1]
        elif i == ei_first and j == ej_first:
            merged = r_i[::-1] + r_j
        elif i == ei_first and j == ej_last:
            merged = r_j + r_i
        else:
            return  # shouldn't happen

        # Update data structures
        routes[ri] = merged
        route_load[ri] = route_load[ri] + route_load[rj]
        route_ends[ri] = (merged[0], merged[-1])

        # Reassign all customers from rj to ri
        for c in r_j:
            route_of[c] = ri

        # Remove old route
        del routes[rj]
        del route_load[rj]
        del route_ends[rj]

    # Apply savings merges
    for s_val, i, j in savings:
        if s_val <= 0:
            break

        ri = route_of[i]
        rj = route_of[j]

        if _can_merge(ri, rj, i, j):
            _merge(ri, rj, i, j)

    result_routes = list(routes.values())

    # If we have more routes than vehicles, try forced merges (smallest first)
    if len(result_routes) > num_vehicles:
        result_routes.sort(key=lambda r: sum(demands[c] for c in r))
        while len(result_routes) > num_vehicles:
            smallest = result_routes.pop(0)
            smallest_load = sum(demands[c] for c in smallest)
            merged = False
            for idx, route in enumerate(result_routes):
                if sum(demands[c] for c in route) + smallest_load <= vehicle_capacity:
                    # Find best insertion position
                    best_cost = np.inf
                    best_pos = 0
                    full_route = [depot] + route + [depot]
                    for c in smallest:
                        for pos in range(1, len(full_route)):
                            cost = (distance_matrix[full_route[pos - 1], c]
                                    + distance_matrix[c, full_route[pos]]
                                    - distance_matrix[full_route[pos - 1], full_route[pos]])
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = pos
                        full_route.insert(best_pos, c)
                    result_routes[idx] = [c for c in full_route if c != depot]
                    merged = True
                    break
            if not merged:
                # Can't merge, put it back
                result_routes.insert(0, smallest)
                break

    return result_routes

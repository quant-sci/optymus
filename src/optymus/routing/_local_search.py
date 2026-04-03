import numpy as np

from ._utils import route_distance


def two_opt_intra(route, distance_matrix, depot=0):
    """Improve a single route by reversing segments (2-opt).

    Repeatedly applies the best improving reversal until no improvement is found.

    Parameters
    ----------
    route : list[int]
        Customer indices (excluding depot).
    distance_matrix : np.ndarray
    depot : int

    Returns
    -------
    list[int]
        Improved route.
    """
    if len(route) < 2:
        return route

    route = list(route)
    improved = True

    while improved:
        improved = False
        full = [depot] + route + [depot]
        n = len(full)

        best_gain = 0.0
        best_i = -1
        best_k = -1

        for i in range(n - 2):
            for k in range(i + 2, n - 1):
                gain = (distance_matrix[full[i], full[i + 1]]
                        + distance_matrix[full[k], full[k + 1]]
                        - distance_matrix[full[i], full[k]]
                        - distance_matrix[full[i + 1], full[k + 1]])
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_k = k

        if best_gain > 1e-10:
            full[best_i + 1:best_k + 1] = full[best_i + 1:best_k + 1][::-1]
            route = full[1:-1]
            improved = True

    return route


def or_opt_intra(route, distance_matrix, depot=0):
    """Improve a single route by relocating chains of 1-3 consecutive customers.

    Parameters
    ----------
    route : list[int]
    distance_matrix : np.ndarray
    depot : int

    Returns
    -------
    list[int]
    """
    if len(route) < 3:
        return route

    route = list(route)
    improved = True

    while improved:
        improved = False
        best_gain = 0.0
        best_move = None

        for seg_len in (1, 2, 3):
            for i in range(len(route) - seg_len + 1):
                seg = route[i:i + seg_len]
                full = [depot] + route + [depot]
                fi = i + 1  # position in full

                # Cost of removing segment
                removal_cost = (distance_matrix[full[fi - 1], full[fi]]
                                + distance_matrix[full[fi + seg_len - 1], full[fi + seg_len]]
                                - distance_matrix[full[fi - 1], full[fi + seg_len]])

                # Try inserting segment at every other position
                remaining = route[:i] + route[i + seg_len:]
                full_rem = [depot] + remaining + [depot]

                for j in range(len(full_rem) - 1):
                    insert_cost = (distance_matrix[full_rem[j], seg[0]]
                                   + distance_matrix[seg[-1], full_rem[j + 1]]
                                   - distance_matrix[full_rem[j], full_rem[j + 1]])
                    gain = removal_cost - insert_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (i, seg_len, j)

        if best_gain > 1e-10 and best_move is not None:
            i, seg_len, j = best_move
            seg = route[i:i + seg_len]
            remaining = route[:i] + route[i + seg_len:]
            remaining.insert(j, None)  # placeholder
            idx = remaining.index(None)
            remaining = remaining[:idx] + seg + remaining[idx + 1:]
            route = remaining
            improved = True

    return route


def relocate_inter(routes, distance_matrix, demands, vehicle_capacity, depot=0):
    """Improve solution by relocating customers between routes.

    For each customer, tries moving it from its current route to the best
    insertion position in another route (if capacity allows and cost improves).

    Parameters
    ----------
    routes : list[list[int]]
    distance_matrix : np.ndarray
    demands : np.ndarray
    vehicle_capacity : float
    depot : int

    Returns
    -------
    list[list[int]]
    """
    improved = True
    routes = [list(r) for r in routes]
    loads = [sum(demands[c] for c in r) for r in routes]

    while improved:
        improved = False

        for ri in range(len(routes)):
            for ci_pos in range(len(routes[ri])):
                c = routes[ri][ci_pos]
                c_demand = demands[c]

                full_ri = [depot] + routes[ri] + [depot]
                fp = ci_pos + 1
                removal_gain = (distance_matrix[full_ri[fp - 1], full_ri[fp]]
                                + distance_matrix[full_ri[fp], full_ri[fp + 1]]
                                - distance_matrix[full_ri[fp - 1], full_ri[fp + 1]])

                best_net_gain = 0.0
                best_rj = -1
                best_insert_pos = -1

                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    if loads[rj] + c_demand > vehicle_capacity:
                        continue

                    full_rj = [depot] + routes[rj] + [depot]
                    for pos in range(1, len(full_rj)):
                        insert_cost = (distance_matrix[full_rj[pos - 1], c]
                                       + distance_matrix[c, full_rj[pos]]
                                       - distance_matrix[full_rj[pos - 1], full_rj[pos]])
                        net_gain = removal_gain - insert_cost
                        if net_gain > best_net_gain:
                            best_net_gain = net_gain
                            best_rj = rj
                            best_insert_pos = pos - 1

                if best_net_gain > 1e-10:
                    routes[ri].pop(ci_pos)
                    loads[ri] -= c_demand
                    routes[best_rj].insert(best_insert_pos, c)
                    loads[best_rj] += c_demand
                    improved = True
                    break
            if improved:
                break

    routes = [r for r in routes if r]
    return routes


def swap_inter(routes, distance_matrix, demands, vehicle_capacity, depot=0):
    """Improve solution by swapping customers between routes.

    For each pair of routes, tries swapping customer c1 from route A with
    customer c2 from route B, accepting the best improving swap.

    Parameters
    ----------
    routes : list[list[int]]
    distance_matrix : np.ndarray
    demands : np.ndarray
    vehicle_capacity : float
    depot : int

    Returns
    -------
    list[list[int]]
    """
    improved = True
    routes = [list(r) for r in routes]
    loads = [sum(demands[c] for c in r) for r in routes]

    while improved:
        improved = False
        best_gain = 0.0
        best_ri = best_rj = best_pi = best_pj = -1

        for ri in range(len(routes)):
            full_ri = [depot] + routes[ri] + [depot]
            for pi in range(len(routes[ri])):
                c1 = routes[ri][pi]
                fp_i = pi + 1

                for rj in range(ri + 1, len(routes)):
                    full_rj = [depot] + routes[rj] + [depot]
                    for pj in range(len(routes[rj])):
                        c2 = routes[rj][pj]

                        new_load_ri = loads[ri] - demands[c1] + demands[c2]
                        new_load_rj = loads[rj] - demands[c2] + demands[c1]
                        if new_load_ri > vehicle_capacity or new_load_rj > vehicle_capacity:
                            continue

                        fp_j = pj + 1
                        old_cost = (distance_matrix[full_ri[fp_i - 1], c1]
                                    + distance_matrix[c1, full_ri[fp_i + 1]]
                                    + distance_matrix[full_rj[fp_j - 1], c2]
                                    + distance_matrix[c2, full_rj[fp_j + 1]])
                        new_cost = (distance_matrix[full_ri[fp_i - 1], c2]
                                    + distance_matrix[c2, full_ri[fp_i + 1]]
                                    + distance_matrix[full_rj[fp_j - 1], c1]
                                    + distance_matrix[c1, full_rj[fp_j + 1]])
                        gain = old_cost - new_cost
                        if gain > best_gain:
                            best_gain = gain
                            best_ri, best_rj = ri, rj
                            best_pi, best_pj = pi, pj

        if best_gain > 1e-10:
            c1 = routes[best_ri][best_pi]
            c2 = routes[best_rj][best_pj]
            routes[best_ri][best_pi] = c2
            routes[best_rj][best_pj] = c1
            loads[best_ri] = loads[best_ri] - demands[c1] + demands[c2]
            loads[best_rj] = loads[best_rj] - demands[c2] + demands[c1]
            improved = True

    return routes


def improve_solution(routes, distance_matrix, demands, vehicle_capacity,
                     depot=0, max_iter=100):
    """Apply local search improvements iteratively.

    Applies intra-route 2-opt, or-opt, inter-route relocations, and
    inter-route swaps until no improvement is found.

    Parameters
    ----------
    routes : list[list[int]]
    distance_matrix : np.ndarray
    demands : np.ndarray
    vehicle_capacity : float
    depot : int
    max_iter : int
        Maximum number of improvement rounds.

    Returns
    -------
    routes : list[list[int]]
    num_iter : int
        Number of rounds actually executed.
    """
    routes = [list(r) for r in routes]
    best_cost = sum(route_distance(r, distance_matrix, depot) for r in routes)

    for iteration in range(1, max_iter + 1):
        # Intra-route improvements
        routes = [two_opt_intra(r, distance_matrix, depot) for r in routes]
        routes = [or_opt_intra(r, distance_matrix, depot) for r in routes]

        # Inter-route improvements
        routes = relocate_inter(routes, distance_matrix, demands, vehicle_capacity, depot)
        routes = swap_inter(routes, distance_matrix, demands, vehicle_capacity, depot)

        # Re-optimize routes after inter-route moves
        routes = [two_opt_intra(r, distance_matrix, depot) for r in routes]

        current_cost = sum(route_distance(r, distance_matrix, depot) for r in routes)

        if current_cost < best_cost - 1e-10:
            best_cost = current_cost
        else:
            return routes, iteration

    return routes, max_iter

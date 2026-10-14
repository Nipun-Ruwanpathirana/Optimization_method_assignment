"""Clarke-Wright Savings heuristic for the Capacitated Vehicle Routing Problem.

The algorithm begins with every customer served on an individual back-and-forth
route [0, i, 0]. Savings s_ij = c_0i + c_0j - lambda * c_ij are computed for every
pair of customers. Customer pairs are processed in descending order of savings,
merging routes whenever capacity and vehicle constraints allow.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from src.checker import check_solution, route_cost, solution_cost
from src.data_loader import Instance


def clarke_wright_savings(
    inst: Instance,
    savings_lambda: float = 1.0,
) -> list[list[int]]:
    """Construct an initial feasible CVRP solution using Clarke-Wright Savings.

    Args:
        inst: The CVRP instance to solve.
        savings_lambda: Weight on customer-to-customer distance (standard = 1.0).

    Returns:
        A list of routes, where each route is a list of customer IDs (depot 0 omitted).
    """
    n = inst.n_customers
    customers = list(inst.customers)

    # Initial state: each customer on their own dedicated route
    routes: dict[int, list[int]] = {i: [i] for i in customers}
    node_to_route: dict[int, int] = {i: i for i in customers}
    route_loads: dict[int, int] = {i: int(inst.demand[i]) for i in customers}

    # Compute savings s_ij = dist[0, i] + dist[0, j] - lambda * dist[i, j]
    savings: list[tuple[float, int, int]] = []
    for idx_i in range(n):
        i = customers[idx_i]
        c_0i = inst.dist[0, i]
        for idx_j in range(idx_i + 1, n):
            j = customers[idx_j]
            c_0j = inst.dist[0, j]
            c_ij = inst.dist[i, j]
            s = float(c_0i + c_0j - savings_lambda * c_ij)
            savings.append((s, i, j))

    # Sort pairs by descending savings
    savings.sort(key=lambda item: item[0], reverse=True)

    # Iterative merge phase
    for _, i, j in savings:
        r_i = node_to_route[i]
        r_j = node_to_route[j]

        # Cannot merge if already on the same route
        if r_i == r_j:
            continue

        route_i = routes[r_i]
        route_j = routes[r_j]

        # Capacity check
        combined_load = route_loads[r_i] + route_loads[r_j]
        if combined_load > inst.capacity:
            continue

        # Feasibility check: i and j must be at the exterior of their respective routes
        i_is_start = (route_i[0] == i)
        i_is_end = (route_i[-1] == i)
        j_is_start = (route_j[0] == j)
        j_is_end = (route_j[-1] == j)

        if not (i_is_start or i_is_end) or not (j_is_start or j_is_end):
            continue

        # Merge orientations
        if i_is_end and j_is_start:
            merged = route_i + route_j
        elif i_is_start and j_is_end:
            merged = route_j + route_i
        elif i_is_end and j_is_end:
            merged = route_i + route_j[::-1]
        elif i_is_start and j_is_start:
            merged = route_i[::-1] + route_j
        else:
            continue

        # Commit merge to r_i, remove r_j
        routes[r_i] = merged
        route_loads[r_i] = combined_load
        for node in route_j:
            node_to_route[node] = r_i
        del routes[r_j]
        del route_loads[r_j]

    solution = [r for r in routes.values() if r]

    # If fleet size constraint K is exceeded, perform route consolidation or fallback packing
    if inst.max_vehicles is not None and len(solution) > inst.max_vehicles:
        solution = _reduce_fleet_size(inst, solution, inst.max_vehicles)

    return solution


def _reduce_fleet_size(
    inst: Instance,
    routes: list[list[int]],
    max_k: int,
) -> list[list[int]]:
    """Consolidate shortest routes into others to satisfy fleet size K."""
    routes = [list(r) for r in routes]

    # Phase 1: Try greedy cheapest insertion to dissolve lightest routes
    while len(routes) > max_k:
        routes.sort(key=lambda r: sum(inst.demand[c] for c in r))
        dissolved = False

        for r_idx in range(len(routes)):
            candidate_route = routes[r_idx]
            other_routes = [r for i, r in enumerate(routes) if i != r_idx]

            backup = [list(r) for r in other_routes]
            can_fit_all = True

            for cust in candidate_route:
                d_c = inst.demand[cust]
                best_insert: Optional[tuple[int, int, int]] = None
                best_cost_delta = float("inf")

                for target_idx, t_route in enumerate(backup):
                    t_load = sum(inst.demand[c] for c in t_route)
                    if t_load + d_c > inst.capacity:
                        continue

                    path = [0, *t_route, 0]
                    for pos in range(len(t_route) + 1):
                        u, v = path[pos], path[pos + 1]
                        delta = inst.dist[u, cust] + inst.dist[cust, v] - inst.dist[u, v]
                        if delta < best_cost_delta:
                            best_cost_delta = delta
                            best_insert = (target_idx, pos, delta)

                if best_insert is None:
                    can_fit_all = False
                    break
                else:
                    target_idx, pos, _ = best_insert
                    backup[target_idx].insert(pos, cust)

            if can_fit_all:
                routes = backup
                dissolved = True
                break

        if not dissolved:
            break

    # Phase 2: If still > max_k (e.g. extremely tight capacity like P-n23-k8 with 98% tightness),
    # find a feasible K-bin packing partition and construct routes using nearest neighbor
    if len(routes) > max_k:
        packed_bins = _solve_bin_packing_partition(inst, max_k)
        if packed_bins is not None:
            routes = [_build_nearest_neighbor_route(inst, b) for b in packed_bins if b]

    return routes


def _solve_bin_packing_partition(
    inst: Instance,
    k_bins: int,
) -> Optional[list[list[int]]]:
    """Find a partition of all customers into at most k_bins respecting capacity."""
    custs = list(inst.customers)
    demands = {c: int(inst.demand[c]) for c in custs}
    sorted_custs = sorted(custs, key=lambda c: demands[c], reverse=True)

    bins: list[list[int]] = [[] for _ in range(k_bins)]
    bin_loads: list[int] = [0] * k_bins

    def backtrack(idx: int) -> bool:
        if idx == len(sorted_custs):
            return True
        c = sorted_custs[idx]
        d = demands[c]
        seen_loads = set()
        for b in range(k_bins):
            if bin_loads[b] + d <= inst.capacity and bin_loads[b] not in seen_loads:
                seen_loads.add(bin_loads[b])
                bin_loads[b] += d
                bins[b].append(c)
                if backtrack(idx + 1):
                    return True
                bins[b].pop()
                bin_loads[b] -= d
        return False

    if backtrack(0):
        return bins
    return None


def _build_nearest_neighbor_route(inst: Instance, nodes: list[int]) -> list[int]:
    """Build a tour for a cluster of nodes using Nearest Neighbor starting from depot."""
    if not nodes:
        return []
    remaining = set(nodes)
    current = 0
    route = []
    while remaining:
        nxt = min(remaining, key=lambda node: inst.dist[current, node])
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return route

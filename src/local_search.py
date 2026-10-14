"""Local search operators for the Capacitated Vehicle Routing Problem.

Implements three classical CVRP neighborhoods:
1. Intra-route 2-Opt: Reverses path subsegments to untangle crossed edges.
2. Inter-route Relocate (1-0 shift): Moves a customer from one route to another.
3. Inter-route Swap (1-1 exchange): Swaps two customers between different routes.

A Variable Neighborhood Descent (VND) engine coordinates them to reach a local optimum.
"""
from __future__ import annotations

import copy
from typing import Optional

from src.checker import route_cost, solution_cost
from src.data_loader import Instance


def two_opt_route(inst: Instance, route: list[int]) -> tuple[list[int], int]:
    """Improve a single route using 2-opt until no improving reversal exists.

    Returns:
        (improved_route, cost_reduction)
    """
    if len(route) < 3:
        return route, 0

    best_route = list(route)
    improved = True
    total_gain = 0

    while improved:
        improved = False
        n = len(best_route)
        # Full path with depot at index 0 and index n+1
        path = [0, *best_route, 0]

        best_delta = 0
        best_i, best_j = -1, -1

        for i in range(1, n):
            for j in range(i + 1, n + 1):
                # We consider reversing the segment from index i to j (1-indexed in path)
                # Old edges: (path[i-1], path[i]) and (path[j], path[j+1])
                # New edges: (path[i-1], path[j]) and (path[i], path[j+1])
                u, v = path[i - 1], path[i]
                x, y = path[j], path[j + 1]

                delta = (inst.dist[u, x] + inst.dist[v, y]) - (inst.dist[u, v] + inst.dist[x, y])
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_delta < 0:
            # Reversal in best_route (0-indexed: i-1 to j-1)
            best_route[best_i - 1 : best_j] = reversed(best_route[best_i - 1 : best_j])
            total_gain -= best_delta
            improved = True

    return best_route, total_gain


def apply_2opt_all(inst: Instance, routes: list[list[int]]) -> list[list[int]]:
    """Apply intra-route 2-opt to all routes in the solution."""
    return [two_opt_route(inst, r)[0] for r in routes if r]


def relocate_move(
    inst: Instance,
    routes: list[list[int]],
) -> tuple[Optional[list[list[int]]], int]:
    """Find the best feasible inter-route Relocate (1-0 shift) move.

    Moves customer c from route r1 to an insertion position in route r2.
    Returns:
        (new_routes, delta_cost) or (None, 0) if no improving move found.
    """
    routes = [list(r) for r in routes if r]
    loads = [int(sum(inst.demand[c] for c in r)) for r in routes]
    costs = [route_cost(inst, r) for r in routes]

    best_delta = 0
    best_move: Optional[tuple[int, int, int, int]] = None  # (src_idx, cust_pos, dst_idx, insert_pos)

    for src_idx, src_route in enumerate(routes):
        if len(src_route) == 0:
            continue

        for cust_pos, cust in enumerate(src_route):
            demand_c = inst.demand[cust]

            # Cost change in src_route if cust is removed
            # src path: [0, ... path ... 0]
            p_src = [0, *src_route, 0]
            u_prev = p_src[cust_pos]
            u_next = p_src[cust_pos + 2]
            src_delta = inst.dist[u_prev, u_next] - (inst.dist[u_prev, cust] + inst.dist[cust, u_next])

            for dst_idx, dst_route in enumerate(routes):
                if dst_idx == src_idx:
                    continue

                # Capacity check for destination
                if loads[dst_idx] + demand_c > inst.capacity:
                    continue

                p_dst = [0, *dst_route, 0]
                for ins_pos in range(len(dst_route) + 1):
                    v_prev = p_dst[ins_pos]
                    v_next = p_dst[ins_pos + 1]
                    dst_delta = (inst.dist[v_prev, cust] + inst.dist[cust, v_next]) - inst.dist[v_prev, v_next]

                    total_delta = src_delta + dst_delta
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_move = (src_idx, cust_pos, dst_idx, ins_pos)

    if best_move is not None:
        src_idx, cust_pos, dst_idx, ins_pos = best_move
        new_routes = [list(r) for r in routes]
        c = new_routes[src_idx].pop(cust_pos)
        new_routes[dst_idx].insert(ins_pos, c)
        new_routes = [r for r in new_routes if r]
        return new_routes, best_delta

    return None, 0


def swap_move(
    inst: Instance,
    routes: list[list[int]],
) -> tuple[Optional[list[list[int]]], int]:
    """Find the best feasible inter-route Swap (1-1 exchange) move.

    Exchanges customer u in route r1 with customer v in route r2.
    Returns:
        (new_routes, delta_cost) or (None, 0) if no improving move found.
    """
    routes = [list(r) for r in routes if r]
    loads = [int(sum(inst.demand[c] for c in r)) for r in routes]
    costs = [route_cost(inst, r) for r in routes]

    best_delta = 0
    best_move: Optional[tuple[int, int, int, int]] = None  # (r1_idx, pos1, r2_idx, pos2)

    for r1_idx in range(len(routes)):
        for r2_idx in range(r1_idx + 1, len(routes)):
            r1, r2 = routes[r1_idx], routes[r2_idx]

            for pos1, u in enumerate(r1):
                d_u = inst.demand[u]
                for pos2, v in enumerate(r2):
                    d_v = inst.demand[v]

                    # Capacity check
                    if loads[r1_idx] - d_u + d_v > inst.capacity:
                        continue
                    if loads[r2_idx] - d_v + d_u > inst.capacity:
                        continue

                    # Evaluate new costs
                    new_r1 = list(r1)
                    new_r2 = list(r2)
                    new_r1[pos1] = v
                    new_r2[pos2] = u

                    delta = (route_cost(inst, new_r1) + route_cost(inst, new_r2)) - (costs[r1_idx] + costs[r2_idx])
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (r1_idx, pos1, r2_idx, pos2)

    if best_move is not None:
        r1_idx, pos1, r2_idx, pos2 = best_move
        new_routes = [list(r) for r in routes]
        u = new_routes[r1_idx][pos1]
        v = new_routes[r2_idx][pos2]
        new_routes[r1_idx][pos1] = v
        new_routes[r2_idx][pos2] = u
        return new_routes, best_delta

    return None, 0


def run_local_search(
    inst: Instance,
    routes: list[list[int]],
    max_passes: int = 50,
) -> list[list[int]]:
    """Run Variable Neighborhood Descent combining 2-Opt, Relocate, and Swap until local optimum."""
    current_routes = apply_2opt_all(inst, routes)

    for _ in range(max_passes):
        # 1. Try best relocate
        improved_routes, delta = relocate_move(inst, current_routes)
        if improved_routes is not None and delta < 0:
            current_routes = apply_2opt_all(inst, improved_routes)
            continue

        # 2. Try best swap
        improved_routes, delta = swap_move(inst, current_routes)
        if improved_routes is not None and delta < 0:
            current_routes = apply_2opt_all(inst, improved_routes)
            continue

        # If neither relocate nor swap can improve, we are in a local optimum
        break

    return apply_2opt_all(inst, current_routes)

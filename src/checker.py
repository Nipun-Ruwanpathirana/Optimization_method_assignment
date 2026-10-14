"""Independent feasibility checker shared by the exact method and the heuristic.

Never trust the cost a solver or heuristic reports about itself: every solution
is re-checked here against the original CVRP rules and its cost is recomputed.
"""
from collections import Counter
from dataclasses import dataclass, field

from src.data_loader import Instance


@dataclass
class CheckResult:
    feasible: bool
    cost: int
    n_routes: int
    loads: list[int]
    errors: list[str] = field(default_factory=list)


def route_cost(inst: Instance, route: list[int]) -> int:
    """Distance of depot -> route[0] -> ... -> route[-1] -> depot."""
    path = [0, *route, 0]
    return int(sum(inst.dist[a, b] for a, b in zip(path, path[1:])))


def solution_cost(inst: Instance, routes: list[list[int]]) -> int:
    return sum(route_cost(inst, r) for r in routes)


def check_solution(inst: Instance, routes: list[list[int]], claimed_cost: float | None = None) -> CheckResult:
    errors = []
    routes = [list(r) for r in routes]

    # Rule: every route starts and ends at the depot, so it must not visit node 0 itself.
    for k, route in enumerate(routes, start=1):
        if not route:
            errors.append(f"route {k} is empty")
        bad_nodes = [v for v in route if v not in inst.customers]
        if bad_nodes:
            errors.append(f"route {k} visits nodes that are not customers: {bad_nodes}")

    # Rule: each customer is served exactly once.
    visits = Counter(v for route in routes for v in route)
    missing = [i for i in inst.customers if visits[i] == 0]
    repeated = sorted(i for i, count in visits.items() if count > 1)
    if missing:
        errors.append(f"customers never visited: {missing}")
    if repeated:
        errors.append(f"customers visited more than once: {repeated}")

    # Rule: total demand on a route <= Q.
    loads = [int(sum(inst.demand[v] for v in route if v in inst.customers)) for route in routes]
    for k, load in enumerate(loads, start=1):
        if load > inst.capacity:
            errors.append(f"route {k} load {load} exceeds capacity {inst.capacity}")

    # Rule: at most K vehicles.
    if inst.max_vehicles is not None and len(routes) > inst.max_vehicles:
        errors.append(f"{len(routes)} routes used but only {inst.max_vehicles} vehicles available")

    cost = solution_cost(inst, [[v for v in r if v in inst.customers] for r in routes])
    if claimed_cost is not None and abs(cost - claimed_cost) > 1e-6:
        errors.append(f"claimed cost {claimed_cost} but recomputed cost is {cost}")

    return CheckResult(feasible=not errors, cost=cost, n_routes=len(routes), loads=loads, errors=errors)

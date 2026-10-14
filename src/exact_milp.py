"""Exact method: CVRP as a Mixed-Integer Linear Program, solved by Branch-and-Bound/Cut (HiGHS).

Two-index vehicle-flow model with Miller-Tucker-Zemlin (MTZ) load variables:

    min   sum_{i,j} c_ij x_ij                                                   (1)
    s.t.  sum_j x_ij = 1                       for every customer i            (2)
          sum_i x_ij = 1                       for every customer j            (3)
          sum_j x_0j <= K                      (fleet size)                    (4)
          sum_j x_0j  = sum_i x_i0             (every route returns)           (5)
          u_j >= u_i + q_j - Q (1 - x_ij)      for customers i != j            (6)
          q_i <= u_i <= Q                      for every customer i            (7)
          x_ij in {0,1}

x_ij = 1 if a vehicle drives directly from node i to node j; u_i = load delivered
on the route up to and including customer i. Constraint (6) both propagates the
load and forbids sub-tours that do not pass through the depot.

`strengthened=True` adds two valid inequalities that tighten the LP relaxation:
    (4b) sum_j x_0j >= ceil(sum q / Q)                    (minimum number of routes)
    (6') lifted MTZ (Kara, Laporte & Bektas, 2004) replacing (6):
         u_i - u_j + Q x_ij + (Q - q_i - q_j) x_ji <= Q - q_j

Usage:
    python -m src.exact_milp P-n16-k8 --time-limit 60
"""
import argparse
import math
import time
from dataclasses import dataclass, field

import highspy
import pulp

from src.checker import check_solution
from src.data_loader import Instance, load_instance


@dataclass
class ExactResult:
    instance: str
    model: str               # "basic" or "strengthened"
    status: str              # optimal | time_limit | no_solution | infeasible | error
    cost: float | None       # best integer solution found (incumbent), recomputed by the checker
    bound: float | None      # best lower bound proven by branch-and-bound
    gap: float | None        # 100 * (cost - bound) / cost
    runtime: float           # wall-clock seconds
    cpu_time: float          # CPU seconds of this process (a sanity check on runtime)
    nodes: int | None
    n_binary: int
    n_continuous: int
    n_constraints: int
    feasible: bool = False
    routes: list[list[int]] = field(default_factory=list)


def build_model(inst: Instance, strengthened: bool = False, relax: bool = False):
    """Return (problem, x, u). With relax=True the binaries become 0 <= x <= 1 (LP relaxation)."""
    V = range(len(inst.demand))
    N = list(inst.customers)
    Q, q, c = inst.capacity, inst.demand, inst.dist

    prob = pulp.LpProblem(f"CVRP_{inst.name}", pulp.LpMinimize)
    arcs = [(i, j) for i in V for j in V if i != j]
    x = pulp.LpVariable.dicts("x", arcs, lowBound=0, upBound=1, cat="Continuous" if relax else "Binary")
    u = {i: pulp.LpVariable(f"u_{i}", lowBound=int(q[i]), upBound=Q) for i in N}          # (7)

    prob += pulp.lpSum(int(c[i, j]) * x[i, j] for (i, j) in arcs)                          # (1)

    for i in N:
        prob += pulp.lpSum(x[i, j] for j in V if j != i) == 1, f"leave_{i}"                # (2)
        prob += pulp.lpSum(x[j, i] for j in V if j != i) == 1, f"enter_{i}"                # (3)

    routes_out = pulp.lpSum(x[0, j] for j in N)
    if inst.max_vehicles is not None:
        prob += routes_out <= inst.max_vehicles, "fleet_size"                               # (4)
    prob += routes_out == pulp.lpSum(x[i, 0] for i in N), "routes_return"                   # (5)

    if strengthened:
        prob += routes_out >= inst.min_vehicles, "min_routes"                               # (4b)

    for i in N:
        for j in N:
            if i == j:
                continue
            if strengthened:                                                                # (6')
                prob += (u[i] - u[j] + Q * x[i, j] + (Q - int(q[i]) - int(q[j])) * x[j, i]
                         <= Q - int(q[j])), f"load_{i}_{j}"
            else:                                                                           # (6)
                prob += u[j] >= u[i] + int(q[j]) - Q * (1 - x[i, j]), f"load_{i}_{j}"
    return prob, x, u


def extract_routes(inst: Instance, x) -> list[list[int]]:
    """Follow the arcs with x_ij = 1 from the depot to rebuild each route."""
    succ = {}
    for (i, j), var in x.items():
        if var.value() is not None and var.value() > 0.5:
            succ.setdefault(i, []).append(j)
    routes = []
    for first in succ.get(0, []):
        route, node, seen = [], first, set()
        while node != 0 and node not in seen:
            route.append(node)
            seen.add(node)
            node = succ.get(node, [0])[0]
        routes.append(route)
    return routes


def _count(prob) -> tuple[int, int, int]:
    variables = prob.variables()
    n_bin = sum(v.cat == pulp.LpInteger for v in variables)
    return n_bin, len(variables) - n_bin, len(prob.constraints)


def solve_exact(inst: Instance, time_limit: float = 300, strengthened: bool = False, msg: bool = False) -> ExactResult:
    prob, x, _ = build_model(inst, strengthened=strengthened)
    n_bin, n_cont, n_cons = _count(prob)

    # Costs are integers, so a proven gap below 1 distance unit already means optimal.
    solver = pulp.HiGHS(msg=msg, timeLimit=time_limit, gapRel=0.0, gapAbs=0.999)
    start, cpu_start = time.perf_counter(), time.process_time()
    prob.solve(solver)
    runtime, cpu_time = time.perf_counter() - start, time.process_time() - cpu_start

    highs = prob.solverModel
    info = highs.getInfo()
    model_status = highs.getModelStatus()
    has_solution = info.primal_solution_status == 2  # 2 = feasible solution available
    bound = info.mip_dual_bound if math.isfinite(info.mip_dual_bound) else None

    if model_status == highspy.HighsModelStatus.kOptimal:
        status = "optimal"
    elif model_status == highspy.HighsModelStatus.kInfeasible:
        status = "infeasible"
    elif model_status == highspy.HighsModelStatus.kTimeLimit:
        status = "time_limit" if has_solution else "no_solution"
    else:
        status = "error"

    result = ExactResult(
        instance=inst.name, model="strengthened" if strengthened else "basic", status=status,
        cost=None, bound=bound, gap=None, runtime=runtime, cpu_time=cpu_time, nodes=int(info.mip_node_count),
        n_binary=n_bin, n_continuous=n_cont, n_constraints=n_cons,
    )
    if has_solution:
        routes = extract_routes(inst, x)
        check = check_solution(inst, routes)
        result.routes, result.cost, result.feasible = routes, check.cost, check.feasible
        if bound is not None:
            result.gap = 100 * max(0.0, check.cost - bound) / check.cost
    return result


def lp_relaxation_bound(inst: Instance, strengthened: bool = False) -> tuple[float, float]:
    """Objective value of the LP relaxation (a lower bound on the optimum) and its solve time."""
    prob, _, _ = build_model(inst, strengthened=strengthened, relax=True)
    start = time.perf_counter()
    prob.solve(pulp.HiGHS(msg=False))
    return pulp.value(prob.objective), time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a CVRPLIB instance exactly with the MTZ MILP.")
    parser.add_argument("name", help="instance name, e.g. P-n16-k8")
    parser.add_argument("--time-limit", type=float, default=300)
    parser.add_argument("--strengthened", action="store_true")
    parser.add_argument("--log", action="store_true", help="show the HiGHS solver log")
    args = parser.parse_args()

    inst = load_instance(args.name)
    lp, _ = lp_relaxation_bound(inst, args.strengthened)
    res = solve_exact(inst, args.time_limit, args.strengthened, msg=args.log)
    print(f"{res.instance} [{res.model}] status={res.status} cost={res.cost} bound={res.bound} "
          f"gap={res.gap if res.gap is None else round(res.gap, 2)}% time={res.runtime:.1f}s "
          f"nodes={res.nodes} LP bound={lp:.1f} feasible={res.feasible}")
    for k, route in enumerate(res.routes, 1):
        print(f"  route {k}: {route}")


if __name__ == "__main__":
    main()

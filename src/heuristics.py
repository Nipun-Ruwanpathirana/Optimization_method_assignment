"""Unified interface for CVRP Heuristics and Metaheuristics.

Provides simple top-level solvers for:
1. Clarke-Wright Savings (Construction)
2. Variable Neighborhood Descent / 2-Opt (Local Search)
3. Simulated Annealing (Metaheuristic)
"""
from __future__ import annotations

import time
from typing import Optional, Union

from src.checker import CheckResult, check_solution, solution_cost
from src.clarke_wright import clarke_wright_savings
from src.data_loader import Instance, load_instance
from src.local_search import apply_2opt_all, run_local_search
from src.simulated_annealing import SAResult, SimulatedAnnealingCVRP


def _resolve_instance(inst_or_name: Union[str, Instance]) -> Instance:
    if isinstance(inst_or_name, str):
        return load_instance(inst_or_name)
    return inst_or_name


def solve_clarke_wright(
    inst_or_name: Union[str, Instance],
    savings_lambda: float = 1.0,
) -> tuple[list[list[int]], int, float, CheckResult]:
    """Solve using Clarke-Wright Savings construction heuristic.

    Returns:
        (routes, cost, runtime_seconds, check_result)
    """
    inst = _resolve_instance(inst_or_name)
    t0 = time.perf_counter()
    routes = clarke_wright_savings(inst, savings_lambda=savings_lambda)
    runtime = time.perf_counter() - t0
    cost = solution_cost(inst, routes)
    check = check_solution(inst, routes, claimed_cost=cost)
    return routes, cost, runtime, check


def solve_local_search(
    inst_or_name: Union[str, Instance],
    initial_routes: Optional[list[list[int]]] = None,
) -> tuple[list[list[int]], int, float, CheckResult]:
    """Solve using Clarke-Wright + Variable Neighborhood Descent (2-Opt, Relocate, Swap).

    Returns:
        (routes, cost, runtime_seconds, check_result)
    """
    inst = _resolve_instance(inst_or_name)
    t0 = time.perf_counter()
    if initial_routes is None:
        init_routes = clarke_wright_savings(inst)
    else:
        init_routes = [list(r) for r in initial_routes]

    routes = run_local_search(inst, init_routes)
    runtime = time.perf_counter() - t0
    cost = solution_cost(inst, routes)
    check = check_solution(inst, routes, claimed_cost=cost)
    return routes, cost, runtime, check


def solve_simulated_annealing(
    inst_or_name: Union[str, Instance],
    cooling_rate: float = 0.985,
    max_iterations: int = 8000,
    seed: int = 42,
) -> SAResult:
    """Solve using Simulated Annealing metaheuristic initialized with Clarke-Wright.

    Returns:
        SAResult with best solution, convergence history, and validation.
    """
    inst = _resolve_instance(inst_or_name)
    sa = SimulatedAnnealingCVRP(
        cooling_rate=cooling_rate,
        max_iterations=max_iterations,
        seed=seed,
    )
    return sa.solve(inst)


def run_heuristic_pipeline(inst_or_name: Union[str, Instance]) -> dict[str, Any]:
    """Run the 3-stage heuristic pipeline on an instance and return a summary dict."""
    inst = _resolve_instance(inst_or_name)

    # 1. Clarke-Wright
    cw_routes, cw_cost, cw_time, cw_check = solve_clarke_wright(inst)

    # 2. Local Search
    ls_routes, ls_cost, ls_time, ls_check = solve_local_search(inst, initial_routes=cw_routes)

    # 3. Simulated Annealing
    sa_res = solve_simulated_annealing(inst)

    return {
        "instance": inst.name,
        "customers": inst.n_customers,
        "cw_cost": cw_cost,
        "cw_time": cw_time,
        "cw_feasible": cw_check.feasible,
        "ls_cost": ls_cost,
        "ls_time": ls_time,
        "ls_feasible": ls_check.feasible,
        "sa_cost": sa_res.best_cost,
        "sa_time": sa_res.runtime_seconds,
        "sa_feasible": sa_res.check_result.feasible,
    }

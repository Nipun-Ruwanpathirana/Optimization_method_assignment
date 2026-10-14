"""Unit tests for Heuristics and Metaheuristics (Clarke-Wright, Local Search, Simulated Annealing).

Ensures that every generated solution is strictly valid according to src/checker.py.
Run with:  python -m pytest tests/test_heuristics.py -v
"""
from __future__ import annotations

import pytest

from src.checker import check_solution
from src.clarke_wright import clarke_wright_savings
from src.data_loader import instance_table, load_bks, load_instance
from src.heuristics import (
    solve_clarke_wright,
    solve_local_search,
    solve_simulated_annealing,
)
from src.local_search import apply_2opt_all, run_local_search
from src.simulated_annealing import SimulatedAnnealingCVRP

# All 16 benchmark instances
INSTANCES = [row["name"] for row in instance_table()]
SMALL_INSTANCES = [row["name"] for row in instance_table() if row["tier"] == "small"]


@pytest.mark.parametrize("name", INSTANCES)
def test_clarke_wright_produces_feasible_solution(name):
    """Clarke-Wright savings must produce a strictly feasible solution on every instance."""
    inst = load_instance(name)
    routes = clarke_wright_savings(inst)
    res = check_solution(inst, routes)
    assert res.feasible, f"{name} Clarke-Wright produced invalid solution: {res.errors}"


@pytest.mark.parametrize("name", SMALL_INSTANCES)
def test_local_search_improves_or_maintains_clarke_wright(name):
    """Local Search should never worsen the initial Clarke-Wright solution."""
    inst = load_instance(name)
    cw_routes = clarke_wright_savings(inst)
    cw_res = check_solution(inst, cw_routes)

    ls_routes = run_local_search(inst, cw_routes)
    ls_res = check_solution(inst, ls_routes)

    assert ls_res.feasible, f"{name} Local search produced invalid solution: {ls_res.errors}"
    assert ls_res.cost <= cw_res.cost, f"{name} LS cost {ls_res.cost} > CW cost {cw_res.cost}"


def test_simulated_annealing_runs_and_improves():
    """Simulated Annealing on P-n16-k8 should run and produce a verified feasible solution."""
    inst = load_instance("P-n16-k8")
    sa = SimulatedAnnealingCVRP(max_iterations=1000, seed=42)
    result = sa.solve(inst)

    assert result.check_result.feasible, f"SA produced invalid solution: {result.check_result.errors}"
    assert result.best_cost <= result.initial_cost
    assert len(result.history.iterations) > 0


def test_reproducibility_with_seed():
    """Two SA runs with the same seed must produce identical best costs."""
    inst = load_instance("P-n16-k8")
    sa1 = SimulatedAnnealingCVRP(max_iterations=500, seed=123)
    res1 = sa1.solve(inst)

    sa2 = SimulatedAnnealingCVRP(max_iterations=500, seed=123)
    res2 = sa2.solve(inst)

    assert res1.best_cost == res2.best_cost

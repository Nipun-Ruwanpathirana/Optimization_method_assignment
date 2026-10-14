"""The exact MILP must find the proven optimum on a small instance, with both formulations."""
import pytest

from src.checker import check_solution
from src.data_loader import load_bks, load_instance
from src.exact_milp import build_model, lp_relaxation_bound, solve_exact


@pytest.mark.parametrize("strengthened", [False, True])
def test_exact_matches_published_optimum(strengthened):
    inst = load_instance("P-n16-k8")
    _, optimum = load_bks("P-n16-k8")
    res = solve_exact(inst, time_limit=120, strengthened=strengthened)
    assert res.status == "optimal"
    assert res.cost == optimum
    assert check_solution(inst, res.routes).feasible


def test_lp_relaxation_is_a_lower_bound_and_strengthening_raises_it():
    inst = load_instance("P-n16-k8")
    _, optimum = load_bks("P-n16-k8")
    basic, _ = lp_relaxation_bound(inst, strengthened=False)
    strong, _ = lp_relaxation_bound(inst, strengthened=True)
    assert basic <= strong <= optimum


def test_model_size_matches_formula():
    inst = load_instance("A-n32-k5")
    n = inst.n_customers
    prob, x, u = build_model(inst)
    assert len(x) == n * (n + 1)            # arcs between all pairs of distinct nodes
    assert len(u) == n
    assert len(prob.constraints) == 2 * n + 2 + n * (n - 1)

"""The checker must accept every published best-known solution and reject broken ones.

Run with:  python -m pytest -q
"""
import pytest

from src.checker import check_solution
from src.data_loader import instance_table, load_bks, load_instance

INSTANCES = [row["name"] for row in instance_table()]
SMALL = "P-n16-k8"


@pytest.mark.parametrize("name", INSTANCES)
def test_best_known_solution_is_feasible_with_published_cost(name):
    inst = load_instance(name)
    routes, cost = load_bks(name)
    result = check_solution(inst, routes, claimed_cost=cost)
    assert result.feasible, result.errors
    assert result.cost == cost


@pytest.mark.parametrize("name", INSTANCES)
def test_instance_table_matches_files(name):
    row = next(r for r in instance_table() if r["name"] == name)
    inst = load_instance(name)
    _, cost = load_bks(name)
    assert inst.n_customers == int(row["customers"])
    assert inst.capacity == int(row["capacity"])
    assert cost == float(row["bks"])


@pytest.fixture
def small():
    inst = load_instance(SMALL)
    routes, _ = load_bks(SMALL)
    return inst, routes


def test_rejects_missing_customer(small):
    inst, routes = small
    broken = [routes[0][1:], *routes[1:]]
    assert not check_solution(inst, broken).feasible


def test_rejects_customer_served_twice(small):
    inst, routes = small
    broken = [routes[0] + [routes[1][0]], *routes[1:]]
    assert not check_solution(inst, broken).feasible


def test_rejects_capacity_violation(small):
    inst, routes = small
    merged = [routes[0] + routes[1], *routes[2:]]
    result = check_solution(inst, merged)
    assert any("exceeds capacity" in e for e in result.errors)


def test_rejects_too_many_vehicles(small):
    inst, routes = small
    split = [[v] for route in routes for v in route]
    result = check_solution(inst, split)
    assert any("vehicles available" in e for e in result.errors)


def test_rejects_wrong_claimed_cost(small):
    inst, routes = small
    _, cost = load_bks(SMALL)
    result = check_solution(inst, routes, claimed_cost=cost - 1)
    assert any("recomputed cost" in e for e in result.errors)


def test_rejects_depot_inside_route(small):
    inst, routes = small
    broken = [[*routes[0], 0], *routes[1:]]
    assert not check_solution(inst, broken).feasible

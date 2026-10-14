"""Simulated Annealing metaheuristic for the Capacitated Vehicle Routing Problem.

Simulated Annealing escapes local minima by accepting worsening moves with
Metropolis probability P = exp(-delta / T), where temperature T decreases
geometrically according to a cooling schedule T_{k+1} = alpha * T_k.
"""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.checker import CheckResult, check_solution, route_cost, solution_cost
from src.clarke_wright import clarke_wright_savings
from src.data_loader import Instance
from src.local_search import apply_2opt_all


@dataclass
class SAHistory:
    iterations: list[int] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    current_costs: list[int] = field(default_factory=list)
    best_costs: list[int] = field(default_factory=list)
    accepted_moves: int = 0
    improving_moves: int = 0


@dataclass
class SAResult:
    best_routes: list[list[int]]
    best_cost: int
    initial_cost: int
    runtime_seconds: float
    iterations_run: int
    history: SAHistory
    check_result: CheckResult


class SimulatedAnnealingCVRP:
    """Simulated Annealing metaheuristic for Capacitated Vehicle Routing."""

    def __init__(
        self,
        t_initial: Optional[float] = None,
        cooling_rate: float = 0.985,
        max_iterations: int = 8000,
        iters_per_temp: int = 15,
        t_min: float = 0.05,
        seed: Optional[int] = 42,
        prob_2opt: float = 0.35,
        prob_relocate: float = 0.40,
        prob_swap: float = 0.25,
    ) -> None:
        self.t_initial = t_initial
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.iters_per_temp = iters_per_temp
        self.t_min = t_min
        self.seed = seed
        self.prob_2opt = prob_2opt
        self.prob_relocate = prob_relocate
        self.prob_swap = prob_swap

    def _auto_initial_temperature(
        self,
        inst: Instance,
        initial_routes: list[list[int]],
        target_acceptance: float = 0.50,
        sample_moves: int = 100,
    ) -> float:
        """Estimate T_0 such that initial worsening moves are accepted with ~target_acceptance rate."""
        current_routes = [list(r) for r in initial_routes if r]
        worsening_deltas = []

        for _ in range(sample_moves):
            cand, delta = self._sample_random_neighbor(inst, current_routes)
            if cand is not None and delta > 0:
                worsening_deltas.append(delta)

        if not worsening_deltas:
            return 50.0

        avg_worsening = float(np.mean(worsening_deltas))
        # P = exp(-avg_delta / T_0) => T_0 = -avg_delta / ln(target_acceptance)
        t_0 = -avg_worsening / math.log(max(1e-4, target_acceptance))
        return max(5.0, float(t_0))

    def _sample_random_neighbor(
        self,
        inst: Instance,
        routes: list[list[int]],
    ) -> tuple[Optional[list[list[int]]], int]:
        """Propose a random neighbor from {2-Opt, Relocate, Swap} satisfying capacity."""
        routes = [list(r) for r in routes if r]
        if not routes:
            return None, 0

        p = random.random()

        # 1. 2-Opt move (intra-route)
        if p < self.prob_2opt:
            r_idx = random.randrange(len(routes))
            route = routes[r_idx]
            if len(route) >= 3:
                n = len(route)
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)

                cand_route = list(route)
                cand_route[i : j + 1] = reversed(cand_route[i : j + 1])

                old_c = route_cost(inst, route)
                new_c = route_cost(inst, cand_route)
                delta = new_c - old_c

                cand_solution = [list(r) for r in routes]
                cand_solution[r_idx] = cand_route
                return cand_solution, delta

        # 2. Relocate move (inter-route or intra-route)
        elif p < (self.prob_2opt + self.prob_relocate):
            if len(routes) == 1 and len(routes[0]) >= 3:
                # Relocate inside the single route
                r = list(routes[0])
                c_idx = random.randrange(len(r))
                val = r.pop(c_idx)
                ins_idx = random.randint(0, len(r))
                r.insert(ins_idx, val)
                delta = route_cost(inst, r) - route_cost(inst, routes[0])
                return [r], delta

            elif len(routes) >= 2:
                src_idx = random.randrange(len(routes))
                dst_idx = random.randrange(len(routes))
                if src_idx != dst_idx and len(routes[src_idx]) > 0:
                    c_pos = random.randrange(len(routes[src_idx]))
                    cust = routes[src_idx][c_pos]
                    d_c = inst.demand[cust]

                    dst_load = sum(inst.demand[c] for c in routes[dst_idx])
                    if dst_load + d_c <= inst.capacity:
                        ins_pos = random.randint(0, len(routes[dst_idx]))

                        new_routes = [list(r) for r in routes]
                        c = new_routes[src_idx].pop(c_pos)
                        new_routes[dst_idx].insert(ins_pos, c)

                        old_cost = route_cost(inst, routes[src_idx]) + route_cost(inst, routes[dst_idx])
                        new_cost = route_cost(inst, new_routes[src_idx]) + route_cost(inst, new_routes[dst_idx])
                        new_routes = [r for r in new_routes if r]
                        return new_routes, (new_cost - old_cost)

        # 3. Swap move (inter-route)
        else:
            if len(routes) >= 2:
                r1_idx, r2_idx = random.sample(range(len(routes)), 2)
                r1, r2 = routes[r1_idx], routes[r2_idx]
                if len(r1) > 0 and len(r2) > 0:
                    pos1 = random.randrange(len(r1))
                    pos2 = random.randrange(len(r2))
                    u, v = r1[pos1], r2[pos2]
                    d_u, d_v = inst.demand[u], inst.demand[v]

                    load1 = sum(inst.demand[c] for c in r1)
                    load2 = sum(inst.demand[c] for c in r2)

                    if (load1 - d_u + d_v <= inst.capacity) and (load2 - d_v + d_u <= inst.capacity):
                        cand_r1 = list(r1)
                        cand_r2 = list(r2)
                        cand_r1[pos1] = v
                        cand_r2[pos2] = u

                        old_c = route_cost(inst, r1) + route_cost(inst, r2)
                        new_c = route_cost(inst, cand_r1) + route_cost(inst, cand_r2)
                        cand_solution = [list(r) for r in routes]
                        cand_solution[r1_idx] = cand_r1
                        cand_solution[r2_idx] = cand_r2
                        return cand_solution, (new_c - old_c)

        return None, 0

    def solve(
        self,
        inst: Instance,
        initial_routes: Optional[list[list[int]]] = None,
    ) -> SAResult:
        """Run Simulated Annealing to find a high-quality CVRP solution.

        Args:
            inst: Instance data.
            initial_routes: Optional warm-start solution (defaults to Clarke-Wright).

        Returns:
            SAResult containing the best found routes, cost, convergence history, and validity.
        """
        start_time = time.perf_counter()

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Initial solution: warm-start with Clarke-Wright Savings
        if initial_routes is None:
            current_routes = clarke_wright_savings(inst)
        else:
            current_routes = [list(r) for r in initial_routes if r]

        current_routes = apply_2opt_all(inst, current_routes)
        current_cost = solution_cost(inst, current_routes)
        initial_cost = current_cost

        best_routes = [list(r) for r in current_routes]
        best_cost = current_cost

        # Determine initial temperature
        if self.t_initial is not None:
            temperature = float(self.t_initial)
        else:
            temperature = self._auto_initial_temperature(inst, current_routes)

        history = SAHistory()
        iteration = 0

        while iteration < self.max_iterations and temperature > self.t_min:
            for _ in range(self.iters_per_temp):
                iteration += 1
                if iteration > self.max_iterations:
                    break

                cand_routes, delta = self._sample_random_neighbor(inst, current_routes)
                if cand_routes is None:
                    continue

                accept = False
                if delta <= 0:
                    accept = True
                    history.improving_moves += 1
                else:
                    # Metropolis acceptance criterion
                    prob = math.exp(-delta / max(1e-6, temperature))
                    if random.random() < prob:
                        accept = True

                if accept:
                    current_routes = cand_routes
                    current_cost += delta
                    history.accepted_moves += 1

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_routes = [list(r) for r in current_routes]

                # Log history periodically (e.g. every 20 iters to save memory)
                if iteration % 20 == 0 or iteration == 1:
                    history.iterations.append(iteration)
                    history.temperatures.append(temperature)
                    history.current_costs.append(current_cost)
                    history.best_costs.append(best_cost)

            # Cool down
            temperature *= self.cooling_rate

        # Final polishing: run 2-opt on all routes of the best solution
        best_routes = apply_2opt_all(inst, best_routes)
        best_cost = solution_cost(inst, best_routes)

        runtime = time.perf_counter() - start_time
        check_res = check_solution(inst, best_routes, claimed_cost=best_cost)

        return SAResult(
            best_routes=best_routes,
            best_cost=best_cost,
            initial_cost=initial_cost,
            runtime_seconds=runtime,
            iterations_run=iteration,
            history=history,
            check_result=check_res,
        )

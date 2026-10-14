"""Run comprehensive benchmarks and experiments for the Heuristic / Metaheuristic pipeline.

Saves:
- results/heuristic_results.csv: Performance metrics on all 16 CVRPLIB instances.
- results/figures/heuristic_gap_comparison.png: Optimality gap comparison across methods and tiers.
- results/figures/heuristic_runtime_scalability.png: Scalability runtime vs customer count.
- results/figures/sa_convergence_study.png: SA cost and temperature progression over iterations.
- results/figures/sa_parameter_tuning.png: Effect of cooling rates on solution quality.
- results/figures/route_comparison_A-n32-k5.png: Visual comparison of BKS vs Simulated Annealing routes.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.checker import check_solution, solution_cost
from src.clarke_wright import clarke_wright_savings
from src.data_loader import instance_table, load_bks, load_instance
from src.heuristics import (
    solve_clarke_wright,
    solve_local_search,
    solve_simulated_annealing,
)
from src.simulated_annealing import SimulatedAnnealingCVRP

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


def run_full_benchmark() -> pd.DataFrame:
    """Run Clarke-Wright, Local Search, and Simulated Annealing on all 16 benchmark instances."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    instances = instance_table()
    records = []

    print(f"Running heuristic benchmarks on {len(instances)} instances...")

    for row in instances:
        name = row["name"]
        tier = row["tier"]
        n_cust = int(row["customers"])
        k_veh = int(row["vehicles"]) if row["vehicles"] else None
        bks = float(row["bks"])

        inst = load_instance(name)
        print(f"  [{tier.upper():6s}] {name:12s} (n={n_cust})...", end="", flush=True)

        # 1. Clarke-Wright
        t0 = time.perf_counter()
        cw_routes = clarke_wright_savings(inst)
        cw_time = time.perf_counter() - t0
        cw_cost = solution_cost(inst, cw_routes)
        cw_chk = check_solution(inst, cw_routes, claimed_cost=cw_cost)
        cw_gap = 100.0 * (cw_cost - bks) / bks

        # 2. Local Search
        t0 = time.perf_counter()
        ls_routes, ls_cost, ls_time, ls_chk = solve_local_search(inst, initial_routes=cw_routes)
        ls_time += cw_time  # total pipeline time
        ls_gap = 100.0 * (ls_cost - bks) / bks

        # 3. Simulated Annealing (scale iterations slightly for large tier)
        sa_iters = 12000 if tier == "large" else 8000
        sa_res = solve_simulated_annealing(inst, max_iterations=sa_iters, cooling_rate=0.985, seed=42)
        sa_cost = sa_res.best_cost
        sa_time = sa_res.runtime_seconds
        sa_gap = 100.0 * (sa_cost - bks) / bks
        sa_chk = sa_res.check_result

        records.append({
            "instance": name,
            "tier": tier,
            "customers": n_cust,
            "vehicles": k_veh,
            "bks_cost": bks,
            "cw_cost": cw_cost,
            "cw_gap_pct": round(cw_gap, 2),
            "cw_time_s": round(cw_time, 4),
            "cw_feasible": cw_chk.feasible,
            "ls_cost": ls_cost,
            "ls_gap_pct": round(ls_gap, 2),
            "ls_time_s": round(ls_time, 4),
            "ls_feasible": ls_chk.feasible,
            "sa_cost": sa_cost,
            "sa_gap_pct": round(sa_gap, 2),
            "sa_time_s": round(sa_time, 4),
            "sa_feasible": sa_chk.feasible,
        })
        print(f" done. CW gap: {cw_gap:5.1f}% | LS gap: {ls_gap:5.1f}% | SA gap: {sa_gap:5.1f}% (SA time: {sa_time:.2f}s)")

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "heuristic_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to {csv_path.relative_to(ROOT)}")
    return df


def plot_gap_comparison(df: pd.DataFrame) -> None:
    """Plot optimality gap % across methods and instance size tiers."""
    fig, ax = plt.subplots(figsize=(12, 6))

    instances = df["instance"].tolist()
    x = np.arange(len(instances))
    width = 0.26

    rects1 = ax.bar(x - width, df["cw_gap_pct"], width, label="Clarke-Wright Savings", color="#6baed6")
    rects2 = ax.bar(x, df["ls_gap_pct"], width, label="Local Search (2-Opt + VND)", color="#3182bd")
    rects3 = ax.bar(x + width, df["sa_gap_pct"], width, label="Simulated Annealing", color="#08519c")

    ax.set_ylabel("Optimality Gap (%) relative to BKS", fontsize=12)
    ax.set_title("CVRP Solution Quality Comparison across Methods and Instance Tiers", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Highlight tier boundaries
    ax.axvline(5.5, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(11.5, color="gray", linestyle=":", alpha=0.7)
    ax.text(2.5, ax.get_ylim()[1] * 0.92, "Small Tier", ha="center", fontweight="semibold")
    ax.text(8.5, ax.get_ylim()[1] * 0.92, "Medium Tier", ha="center", fontweight="semibold")
    ax.text(13.5, ax.get_ylim()[1] * 0.92, "Large Tier", ha="center", fontweight="semibold")

    fig.tight_layout()
    out_path = FIG_DIR / "heuristic_gap_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated {out_path.relative_to(ROOT)}")


def plot_runtime_scalability(df: pd.DataFrame) -> None:
    """Plot runtime scaling as customer count increases."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["customers"], df["cw_time_s"], "o-", label="Clarke-Wright Savings", color="#2ca02c", linewidth=2)
    ax.plot(df["customers"], df["ls_time_s"], "s-", label="Local Search (VND)", color="#ff7f0e", linewidth=2)
    ax.plot(df["customers"], df["sa_time_s"], "^-", label="Simulated Annealing", color="#1f77b4", linewidth=2)

    ax.set_xlabel("Number of Customers (n)", fontsize=12)
    ax.set_ylabel("Runtime (seconds, log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Computational Scalability of Heuristic Methods vs Problem Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = FIG_DIR / "heuristic_runtime_scalability.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated {out_path.relative_to(ROOT)}")


def run_sa_convergence_study(instance_name: str = "A-n32-k5") -> None:
    """Run and plot the convergence profile of Simulated Annealing."""
    inst = load_instance(instance_name)
    sa = SimulatedAnnealingCVRP(cooling_rate=0.985, max_iterations=8000, seed=42)
    res = sa.solve(inst)

    h = res.history
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_cost = "#1f77b4"
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Solution Cost", color=color_cost, fontsize=12)
    ax1.plot(h.iterations, h.current_costs, label="Current Solution Cost", color="#aec7e8", alpha=0.7)
    ax1.plot(h.iterations, h.best_costs, label="Best-So-Far Cost", color=color_cost, linewidth=2.5)
    ax1.tick_params(axis="y", labelcolor=color_cost)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Reference optimal
    _, bks_cost = load_bks(instance_name)
    ax1.axhline(bks_cost, color="green", linestyle="--", linewidth=1.8, label=f"Proven Optimal ({int(bks_cost)})")

    ax2 = ax1.twinx()
    color_temp = "#d62728"
    ax2.set_ylabel("Temperature T", color=color_temp, fontsize=12)
    ax2.plot(h.iterations, h.temperatures, color=color_temp, linestyle=":", linewidth=1.5, label="Temperature T")
    ax2.tick_params(axis="y", labelcolor=color_temp)

    ax1.set_title(f"Simulated Annealing Convergence Profile on {instance_name}", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    out_path = FIG_DIR / "sa_convergence_study.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated {out_path.relative_to(ROOT)}")


def run_parameter_tuning_study(instance_name: str = "A-n32-k5") -> None:
    """Study the effect of the geometric cooling rate alpha on solution quality."""
    inst = load_instance(instance_name)
    cooling_rates = [0.90, 0.95, 0.985, 0.995]
    results = {}

    for alpha in cooling_rates:
        sa = SimulatedAnnealingCVRP(cooling_rate=alpha, max_iterations=10000, seed=42)
        res = sa.solve(inst)
        results[alpha] = res

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d95f02", "#7570b3", "#1b9e77", "#e7298a"]

    for (alpha, res), col in zip(results.items(), colors):
        h = res.history
        gap = 100.0 * (res.best_cost - 784) / 784
        ax.plot(h.iterations, h.best_costs, label=f"alpha = {alpha} (Final: {res.best_cost}, Gap: {gap:.1f}%)", color=col, linewidth=2)

    _, bks_cost = load_bks(instance_name)
    ax.axhline(bks_cost, color="black", linestyle="--", label=f"Optimal BKS ({int(bks_cost)})")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Solution Cost", fontsize=12)
    ax.set_title(f"Sensitivity Analysis: Cooling Rate alpha on {instance_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = FIG_DIR / "sa_parameter_tuning.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated {out_path.relative_to(ROOT)}")


def plot_route_comparison(instance_name: str = "A-n32-k5") -> None:
    """Plot visual route map comparing BKS vs Simulated Annealing."""
    inst = load_instance(instance_name)
    bks_routes, bks_cost = load_bks(instance_name)

    sa_res = solve_simulated_annealing(inst, max_iterations=8000, seed=42)
    sa_routes = sa_res.best_routes
    sa_cost = sa_res.best_cost

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("tab10")

    # Plot BKS
    ax1.scatter(inst.coords[0, 0], inst.coords[0, 1], c="black", s=150, marker="s", zorder=5, label="Depot")
    ax1.scatter(inst.coords[1:, 0], inst.coords[1:, 1], c="gray", s=40, zorder=4, label="Customer")
    for r_i, route in enumerate(bks_routes):
        col = cmap(r_i % 10)
        path = [0, *route, 0]
        pts = inst.coords[path]
        ax1.plot(pts[:, 0], pts[:, 1], color=col, linewidth=1.8, alpha=0.85)
    ax1.set_title(f"Optimal Solution (BKS)\nCost = {int(bks_cost)} | Routes = {len(bks_routes)}", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="upper left")

    # Plot Simulated Annealing
    ax2.scatter(inst.coords[0, 0], inst.coords[0, 1], c="black", s=150, marker="s", zorder=5, label="Depot")
    ax2.scatter(inst.coords[1:, 0], inst.coords[1:, 1], c="gray", s=40, zorder=4, label="Customer")
    for r_i, route in enumerate(sa_routes):
        col = cmap(r_i % 10)
        path = [0, *route, 0]
        pts = inst.coords[path]
        ax2.plot(pts[:, 0], pts[:, 1], color=col, linewidth=1.8, alpha=0.85)
    gap = 100.0 * (sa_cost - bks_cost) / bks_cost
    ax2.set_title(f"Simulated Annealing Solution\nCost = {sa_cost} (Gap: {gap:.1f}%) | Routes = {len(sa_routes)}", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="upper left")

    fig.suptitle(f"Route Structure Comparison on {instance_name}", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = FIG_DIR / f"route_comparison_{instance_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path.relative_to(ROOT)}")


def main() -> None:
    print("=== Running Heuristic & Metaheuristic Benchmarking Suite ===")
    df = run_full_benchmark()

    print("\n=== Generating Scientific Figures for Report ===")
    plot_gap_comparison(df)
    plot_runtime_scalability(df)
    run_sa_convergence_study("A-n32-k5")
    run_parameter_tuning_study("A-n32-k5")
    plot_route_comparison("A-n32-k5")
    print("\nAll benchmarking experiments and figures completed successfully!")


if __name__ == "__main__":
    main()

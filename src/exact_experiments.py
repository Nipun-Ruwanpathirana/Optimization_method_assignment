"""Run the exact-method experiments of the report and save the raw results in results/.

    python -m src.exact_experiments exact                      # E1: MILP (basic + strengthened), small/medium tiers
    python -m src.exact_experiments scaling                    # E2: MILP vs SA on growing sub-instances
    python -m src.exact_experiments scaling --heuristic-only   # E2: re-run only the (fast) SA rows

E1 solves every small and medium instance with both MILP formulations (300 s limit each).
E2 takes the first n customers of A-n32-k5 (n = 6..25) and solves each sub-instance with
both MILP formulations and with the simulated annealing pipeline of src/heuristics.py
(Member 2), so runtime and quality can be compared at exactly the same sizes.

Exact runs are independent, so they run in parallel worker processes (--workers).
Each run measures its own wall-clock time with time.perf_counter() and its CPU time
with time.process_time(). Keep the computer plugged in with the lid open: the script
asks Windows not to sleep, because time spent asleep would be counted as solver time.
"""
import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict

import pandas as pd

from src.data_loader import ROOT, instance_table, load_bks, load_instance, sub_instance
from src.exact_milp import lp_relaxation_bound, solve_exact

RESULTS = ROOT / "results"
SOLUTIONS = RESULTS / "solutions"
SCALING_BASE = "A-n32-k5"
SCALING_SIZES = [6, 8, 10, 12, 14, 16, 18, 20, 22, 25]
SCALING_SEEDS = list(range(1, 11))
SA_METHOD = "Clarke-Wright + SA"


@contextmanager
def keep_awake():
    """Ask Windows not to sleep while experiments run (sleep time would count as runtime)."""
    if sys.platform != "win32":
        yield
        return
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


def _save(df: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / name, index=False)
    print(f"saved results/{name} ({len(df)} rows)")


def _save_routes(key: str, routes: list[list[int]]) -> None:
    SOLUTIONS.mkdir(parents=True, exist_ok=True)
    (SOLUTIONS / f"{key}.json").write_text(json.dumps(routes))


# ----- E1: exact method ------------------------------------------------------------------
def _exact_job(args) -> dict:
    name, tier, strengthened, time_limit = args
    inst = load_instance(name)
    _, bks = load_bks(name)
    lp, lp_time = lp_relaxation_bound(inst, strengthened)
    res = solve_exact(inst, time_limit=time_limit, strengthened=strengthened)
    if res.routes:
        _save_routes(f"exact_{res.model}_{name}", res.routes)
    row = asdict(res)
    del row["routes"]
    row.update(tier=tier, n=inst.n_customers, bks=bks, lp_bound=lp, lp_time=lp_time,
               root_gap=100 * (bks - lp) / bks,
               gap_to_bks=None if res.cost is None else 100 * (res.cost - bks) / bks)
    warning = "  WARNING: wall time far above CPU time (sleep?)" if res.runtime > 1.5 * res.cpu_time + 10 else ""
    print(f"  exact {name:11} {res.model:12} {res.status:11} cost={res.cost} bound={res.bound} "
          f"t={res.runtime:6.1f}s cpu={res.cpu_time:6.1f}s{warning}", flush=True)
    return row


def run_exact(time_limit: float, workers: int) -> None:
    jobs = [(r["name"], r["tier"], strong, time_limit)
            for r in instance_table() if r["tier"] in ("small", "medium")
            for strong in (False, True)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_exact_job, jobs))
    _save(pd.DataFrame(rows), "exact.csv")


# ----- E2: scalability on sub-instances ----------------------------------------------------
def _scaling_exact_job(args) -> dict:
    m, strengthened, time_limit = args
    inst = sub_instance(load_instance(SCALING_BASE), m)
    res = solve_exact(inst, time_limit=time_limit, strengthened=strengthened)
    print(f"  scaling exact n={m:2} {res.model:12} {res.status:11} cost={res.cost} t={res.runtime:6.1f}s", flush=True)
    return {"n": m, "method": f"MILP ({res.model})", "seed": None, "status": res.status,
            "cost": res.cost, "bound": res.bound, "mip_gap": res.gap, "runtime": res.runtime,
            "cpu_time": res.cpu_time, "feasible": res.feasible}


def _scaling_sa_rows() -> list[dict]:
    """Member 2's pipeline (Clarke-Wright -> 2-opt -> simulated annealing) on the same sub-instances."""
    from src.heuristics import solve_simulated_annealing

    base = load_instance(SCALING_BASE)
    rows = []
    for m in SCALING_SIZES:
        inst = sub_instance(base, m)
        for seed in SCALING_SEEDS:
            cpu_start = time.process_time()
            res = solve_simulated_annealing(inst, seed=seed)
            rows.append({"n": m, "method": SA_METHOD, "seed": seed, "status": "heuristic",
                         "cost": res.best_cost, "bound": None, "mip_gap": None,
                         "runtime": res.runtime_seconds, "cpu_time": time.process_time() - cpu_start,
                         "feasible": res.check_result.feasible})
        print(f"  scaling SA n={m:2} done", flush=True)
    return rows


def run_scaling(time_limit: float, workers: int, heuristic_only: bool) -> None:
    if heuristic_only:
        old = pd.read_csv(RESULTS / "scaling.csv")
        rows = old[old.method.str.startswith("MILP")].to_dict("records")
    else:
        jobs = [(m, strong, time_limit) for m in SCALING_SIZES for strong in (False, True)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_scaling_exact_job, jobs))
    rows += _scaling_sa_rows()
    _save(pd.DataFrame(rows), "scaling.csv")


def write_environment(args) -> None:
    from importlib.metadata import version
    env = {"date": time.strftime("%Y-%m-%d %H:%M"), "python": platform.python_version(),
           "platform": platform.platform(), "processor": platform.processor(),
           "logical_cpus": __import__("os").cpu_count(),
           **{pkg: version(pkg) for pkg in ("pulp", "highspy", "numpy", "pandas", "vrplib")},
           "exact_time_limit_s": args.time_limit, "parallel_exact_workers": args.workers}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"environment_{args.experiment}.json").write_text(json.dumps(env, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", choices=["exact", "scaling", "all"])
    parser.add_argument("--time-limit", type=float, default=300, help="seconds per exact MILP solve")
    parser.add_argument("--workers", type=int, default=6, help="parallel processes for exact runs")
    parser.add_argument("--heuristic-only", action="store_true",
                        help="scaling: keep the MILP rows of results/scaling.csv and re-run only the SA rows")
    args = parser.parse_args()
    write_environment(args)
    start = time.perf_counter()
    with keep_awake():
        if args.experiment in ("exact", "all"):
            run_exact(args.time_limit, args.workers)
        if args.experiment in ("scaling", "all"):
            run_scaling(args.time_limit, args.workers, args.heuristic_only)
    print(f"finished in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

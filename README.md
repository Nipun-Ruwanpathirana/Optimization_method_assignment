# Capacitated Delivery Route Planning: Exact MILP vs Simulated Annealing

IT5082 Optimization Methods: Programming Assignment (group of 2)

## The problem

A distribution centre (the **depot**) must deliver goods to *n* customers. Each customer needs a known amount (**demand**), and every delivery truck can carry at most **Q** units. At most **K** trucks are available.

**Decision:** which customers each truck serves, and in what order.
**Goal:** minimise the total distance driven.
**Rules:**
1. Every customer is visited exactly once.
2. Every route starts and ends at the depot.
3. The total demand on a route must not exceed the truck capacity Q.
4. No more than K routes are used.

This is the **Capacitated Vehicle Routing Problem (CVRP)**. It contains the Travelling Salesman Problem as a special case, so it is NP-hard. Exact methods prove optimality but their runtime explodes as *n* grows, while heuristics stay fast without guaranteeing optimality. Measuring that trade-off on real benchmark data is the aim of this project.

## Data

Public benchmark instances from [CVRPLIB](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/instances). Each instance comes with a proven optimal cost, which lets us measure exactly how far each method is from the optimum. Details, conventions and citations are in [data/README.md](data/README.md).

| Tier | Instances | Customers | Purpose |
|---|---|---|---|
| small | P-n16-k8, P-n19-k2, P-n20-k2, P-n22-k2, E-n22-k4, P-n23-k8 | 15-22 | Exact method reaches or proves the optimum |
| medium | B-n31-k5, A-n32-k5, A-n37-k5, P-n40-k5, A-n45-k7, E-n51-k5 | 30-50 | Exact method stops at a time limit; compare gaps |
| large | X-n101-k25, X-n153-k22, X-n200-k36, X-n256-k16 | 100-255 | Heuristic only: scalability |

A first trial of a basic MILP (HiGHS, 120 s limit) proved the optimum of P-n16-k8 in 20 s. On the other small instances it stopped 0-3 % above the optimum without a proof, and on the 30-customer instances about 5 % above. This "exact-method wall" is the trade-off the project measures.

## Methods

| | Method | Library |
|---|---|---|
| Exact | Mixed-Integer Linear Program (two-index vehicle-flow model with MTZ load constraints) solved by Branch-and-Bound / Branch-and-Cut | PuLP + CBC / HiGHS |
| Heuristic | Clarke-Wright savings construction, then local search, then Simulated Annealing | own Python implementation |

Every solution from either method goes through the same independent checker (`src/checker.py`), which re-verifies all four rules and recomputes the cost.

## Setup

Requires Python 3.10+ (tested with 3.13.7 on Windows 11).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m src.download_data     # only needed if data/raw/ is missing
python -m pytest -q             # checker tests: every published solution must pass
```

Then open `notebooks/01_data_exploration.ipynb` in VS Code or Jupyter.

## Repository structure

```
data/
  instances.csv          selected instances (tier, size, capacity, best-known cost)
  raw/                   CVRPLIB .vrp instances and .sol best-known solutions
  README.md              provenance, conventions, citations
src/
  download_data.py       fetches the instances listed in instances.csv
  data_loader.py         Instance class, rounded distance matrix, best-known solutions
  checker.py             independent feasibility + cost check for any solution
  clarke_wright.py       Clarke-Wright savings construction heuristic
  local_search.py        2-opt, relocate, and swap neighborhood operators
  simulated_annealing.py Simulated Annealing metaheuristic with geometric cooling
  heuristics.py          unified facade for heuristic solvers
  benchmark_heuristics.py full benchmark runner across all 16 instances
tests/
  test_checker.py        published solutions accepted, broken ones rejected
  test_heuristics.py     heuristic solutions verified against independent checker
notebooks/
  01_data_exploration.ipynb
  02_heuristic_simulated_annealing.ipynb
results/
  heuristic_results.csv  benchmarks on all 16 instances (cost, gap %, runtime)
  figures/               figures used in the report
```

## Progress

- [x] Choose problem and dataset
- [x] Download script and data loader (CVRPLIB distance convention)
- [x] Independent feasibility checker and tests
- [x] Data exploration notebook
- [ ] Exact method: MILP model, solver runs, LP relaxation bound
- [x] Heuristic: savings construction (Clarke-Wright)
- [x] Heuristic: local search (relocate, swap, 2-opt)
- [x] Heuristic: simulated annealing and parameter study
- [x] Experiment runner, results tables and figures
- [ ] Report (PDF), code appendix, video

## Team

See [members.txt](members.txt).

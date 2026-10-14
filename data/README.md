# Data

## Source

All instances come from **CVRPLIB**, the standard public library of Capacitated Vehicle Routing benchmark instances, maintained at PUC-Rio:
https://galgos.inf.puc-rio.br/cvrplib/index.php/en/instances

`instances.csv` lists the instances we use, with their CVRPLIB download id. `python -m src.download_data` fetches them into `raw/`. The files are tiny (about 80 KB in total), so they are committed for reproducibility.

| File | Content |
|---|---|
| `raw/<name>.vrp` | Instance: node coordinates, customer demands, vehicle capacity (TSPLIB/CVRPLIB format) |
| `raw/<name>.sol` | Best-known solution published on CVRPLIB: the routes and their cost |
| `instances.csv` | Our selection: size tier, customers, vehicles, capacity, best-known cost, whether it is proven optimal |

## Role of the data in this project

The data **instantiates and validates** the optimisation model. No model is trained on it.

* Coordinates give the travel cost matrix `c_ij`.
* Demands, capacity and fleet size give `q_i`, `Q` and `K`.
* The published cost is the **reference value** used to compute each method's gap: `gap % = 100 x (our cost - reference) / reference`.

## Conventions (important for a fair comparison)

* **Distance:** `c_ij = nint(Euclidean distance)`, the TSPLIB rounding `floor(d + 0.5)`. The `vrplib` package returns *unrounded* distances, so `src/data_loader.py` rounds them. With this rounding, every published solution reproduces its published cost exactly (checked by `tests/test_checker.py`).
* **Node numbering:** node 0 is the depot and nodes 1..n are customers, as in the `.sol` files.
* **Fleet size K:** for the classic sets (A, B, E, P), K is the `-kK` part of the name. In the X set the fleet is unlimited, as in Uchoa et al. (2017). For example, the best-known solution of `X-n101-k25` uses 26 routes.

## Citations

* Augerat, P., Belenguer, J.M., Benavent, E., Corberán, A., Naddef, D., Rinaldi, G. (1995). *Computational results with a branch and cut code for the capacitated vehicle routing problem.* Research Report 949-M, Université Joseph Fourier, Grenoble. (Sets A, B, P)
* Christofides, N., Eilon, S. (1969). An algorithm for the vehicle-dispatching problem. *Operational Research Quarterly*, 20(3), 309-318. (Set E)
* Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., Subramanian, A. (2017). New benchmark instances for the Capacitated Vehicle Routing Problem. *European Journal of Operational Research*, 257(3), 845-858. (Set X)

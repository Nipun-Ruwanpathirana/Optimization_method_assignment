"""Load a CVRPLIB instance and build the distance matrix shared by every method.

Node 0 is the depot and nodes 1..n are the customers, so the same index is used
for coordinates, demands and the distance matrix everywhere in the project.
"""
from __future__ import annotations
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vrplib

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INSTANCE_LIST = ROOT / "data" / "instances.csv"


@dataclass
class Instance:
    name: str
    coords: np.ndarray       # (n+1, 2) x/y coordinates, row 0 = depot
    demand: np.ndarray       # (n+1,) demand q_i, demand[0] = 0
    capacity: int            # Q, identical for every vehicle
    max_vehicles: int | None # K, fleet limit (None = unlimited, as in the X set)
    dist: np.ndarray         # (n+1, n+1) integer travel cost c_ij

    @property
    def n_customers(self) -> int:
        return len(self.demand) - 1

    @property
    def customers(self) -> range:
        return range(1, len(self.demand))

    @property
    def min_vehicles(self) -> int:
        """Lower bound on the number of routes: ceil(total demand / Q)."""
        return int(np.ceil(self.demand.sum() / self.capacity))


def tsplib_round(x: np.ndarray) -> np.ndarray:
    """TSPLIB 'nint' rounding. CVRPLIB best-known costs are computed with it."""
    return np.floor(x + 0.5).astype(int)


def fleet_limit(name: str) -> int | None:
    """K for the classic sets is the '-kK' part of the name (e.g. A-n32-k5 -> 5).

    In the X set (Uchoa et al., 2017) the fleet size is not fixed, so there is no limit.
    """
    if name.startswith("X-"):
        return None
    match = re.search(r"-k(\d+)$", name)
    return int(match.group(1)) if match else None


def load_instance(name: str, raw_dir: Path = RAW_DIR) -> Instance:
    data = vrplib.read_instance(raw_dir / f"{name}.vrp")

    if list(data["depot"]) != [0]:
        raise ValueError(f"{name}: expected a single depot at node 0, got {data['depot']}")

    # vrplib returns the exact (unrounded) Euclidean distance for EUC_2D instances;
    # round it the TSPLIB way so our costs are comparable with the published ones.
    if data["edge_weight_type"] == "EUC_2D":
        dist = tsplib_round(data["edge_weight"])
    else:
        dist = np.asarray(data["edge_weight"]).round().astype(int)

    return Instance(
        name=name,
        coords=np.asarray(data.get("node_coord", np.zeros((len(dist), 2)))),
        demand=np.asarray(data["demand"], dtype=int),
        capacity=int(data["capacity"]),
        max_vehicles=fleet_limit(name),
        dist=dist,
    )


def load_bks(name: str, raw_dir: Path = RAW_DIR) -> tuple[list[list[int]], float]:
    """Best-known solution published on CVRPLIB: (routes, cost)."""
    solution = vrplib.read_solution(raw_dir / f"{name}.sol")
    return solution["routes"], solution["cost"]


def instance_table() -> list[dict]:
    """Rows of data/instances.csv (name, cvrplib_id, tier, ...)."""
    with open(INSTANCE_LIST, newline="") as f:
        return list(csv.DictReader(f))

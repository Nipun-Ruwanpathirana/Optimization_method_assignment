"""Turn the raw result files into the tables and figures used in the report.

    python -m src.report_analysis

Reads results/exact.csv and results/scaling.csv (exact method, src/exact_experiments.py) and
results/heuristic_results.csv (heuristics, src/benchmark_heuristics.py), and writes
results/tables/*.md|csv and results/figures/*.png. Every number in the report comes from one
of these tables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import ROOT, instance_table, load_bks, load_instance

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
SCALING_BASE = "A-n32-k5"
SA_METHOD = "Clarke-Wright + SA"

# Colour roles (validated categorical palette, light mode). Colour follows the entity.
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_COLOURS = {"MILP (basic)": "#2a78d6", "MILP (strengthened)": "#eb6834", SA_METHOD: "#1baf7a"}
ROUTE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDINAL_BLUES = ["#86b6ef", "#2a78d6", "#104281"]   # light -> dark for the three heuristic layers
TIER_ORDER = ["small", "medium", "large"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 1.8, "lines.markersize": 6,
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def plot_routes(ax, inst, routes, title) -> None:
    """Draw each route in its own colour; the black square is the depot."""
    xy = inst.coords
    for k, route in enumerate(routes):
        path = [0, *route, 0]
        ax.plot(xy[path, 0], xy[path, 1], color=ROUTE_COLOURS[k % len(ROUTE_COLOURS)], lw=1.4)
        ax.plot(xy[route, 0], xy[route, 1], "o", color=ROUTE_COLOURS[k % len(ROUTE_COLOURS)], ms=4,
                mec="white", mew=0.8)
    ax.plot(*xy[0], "s", color=INK, ms=8, zorder=4)
    ax.set_title(title, loc="left")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _save_fig(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"figure  results/figures/{name}")


def _fmt(value, digits: int = 2) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{digits}f}"
    return str(value)


def to_markdown(df: pd.DataFrame, digits: int | dict = 2) -> str:
    """Pipe table; numeric columns right-aligned, floats with a fixed number of decimals
    (one number for all columns, or a {column: decimals} dict).

    The number of dashes under each header is proportional to the widest cell, which pandoc
    uses as the relative column width, so values never wrap in the Word/PDF report."""
    col_digits = [digits.get(c, 2) if isinstance(digits, dict) else digits for c in df.columns]
    cells = [[_fmt(v, d) for v, d in zip(row, col_digits)] for row in df.itertuples(index=False)]
    widths = []
    for k, col in enumerate(df.columns):
        longest_value = max((len(r[k]) for r in cells), default=0)
        longest_word = max(len(w) for w in str(col).split())
        widths.append(max(longest_value, longest_word, 3) + 2)   # +2 = room for the cell padding
    numeric = [pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) for c in df.columns]
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("-" * w + (":" if is_num else "-") for w, is_num in zip(widths, numeric)) + "|"
    body = ["| " + " | ".join(r) + " |" for r in cells]
    return "\n".join([header, sep, *body]) + "\n"


def _save_table(df: pd.DataFrame, name: str, digits: int = 2) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(to_markdown(df, digits), encoding="utf-8")
    print(f"table   results/tables/{name}.md")


def _status(row) -> str:
    return "proven" if row.status == "optimal" else "time limit"


# ----- tables ------------------------------------------------------------------------------
def instances_table() -> None:
    rows = []
    for r in instance_table():
        inst = load_instance(r["name"])
        _, bks = load_bks(r["name"])
        vehicles = inst.max_vehicles or inst.min_vehicles
        rows.append({"Instance": inst.name, "Tier": r["tier"], "n": inst.n_customers,
                     "K": inst.max_vehicles if inst.max_vehicles else "unlimited", "Q": inst.capacity,
                     "Total demand": int(inst.demand.sum()),
                     "Tightness": round(inst.demand.sum() / (vehicles * inst.capacity), 2),
                     "Optimal cost": int(bks)})
    _save_table(pd.DataFrame(rows), "T1_instances")


def exact_table(ex: pd.DataFrame) -> None:
    """Every E1 run (both models), as shown in notebooks/02_exact_method.ipynb."""
    df = ex.sort_values(["n", "instance", "model"])
    out = pd.DataFrame({
        "Instance": df.instance, "n": df.n, "Model": df.model, "Status": df.status.str.replace("_", " "),
        "Cost": df.cost.astype("Int64"), "Lower bound": df.bound.round(1), "MIP gap %": df.gap.round(2),
        "Gap to opt. %": df.gap_to_bks.round(2), "Time s": df.runtime.round(1), "Nodes": df.nodes.astype("Int64"),
        "LP bound": df.lp_bound.round(1), "Root gap %": df.root_gap.round(1),
    })
    _save_table(out, "T2_exact")


def exact_compact_table(ex: pd.DataFrame) -> None:
    """One row per instance, basic and strengthened side by side (report table)."""
    rows = []
    for name, g in ex.sort_values("n").groupby("instance", sort=False):
        b, s = g[g.model == "basic"].iloc[0], g[g.model == "strengthened"].iloc[0]
        rows.append({"Instance": name, "n": int(b.n), "Optimum": int(b.bks),
                     "Basic cost": int(b.cost), "Basic status": _status(b), "Basic time s": round(b.runtime, 1),
                     "Basic nodes": int(b.nodes), "Str. cost": int(s.cost), "Str. status": _status(s),
                     "Str. time s": round(s.runtime, 1), "Str. nodes": int(s.nodes),
                     "Root gap basic %": round(b.root_gap, 1), "Root gap str. %": round(s.root_gap, 1)})
    _save_table(pd.DataFrame(rows), "T3_exact_compact", digits=1)


def heuristic_table(he: pd.DataFrame) -> None:
    out = pd.DataFrame({
        "Instance": he.instance, "Tier": he.tier, "n": he.customers, "Optimum": he.bks_cost.astype(int),
        "CW cost": he.cw_cost, "CW gap %": he.cw_gap_pct, "LS cost": he.ls_cost, "LS gap %": he.ls_gap_pct,
        "SA cost": he.sa_cost, "SA gap %": he.sa_gap_pct, "SA time s": he.sa_time_s.round(2),
        "Feasible": he.cw_feasible & he.ls_feasible & he.sa_feasible,
    })
    _save_table(out, "T4_heuristic")


def tier_table(ex: pd.DataFrame, he: pd.DataFrame) -> None:
    """Head-to-head per tier: proofs and gaps of the MILP vs the three heuristic layers."""
    rows = []
    for tier in TIER_ORDER:
        h = he[he.tier == tier]
        row = {"Tier": tier, "Instances": len(h)}
        for model, label in (("basic", "Basic"), ("strengthened", "Str.")):
            e = ex[(ex.tier == tier) & (ex.model == model)]
            row[f"{label} proven"] = f"{(e.status == 'optimal').sum()}/{len(e)}" if len(e) else "not run"
            row[f"{label} gap %"] = round(e.gap_to_bks.mean(), 2) if len(e) else None
            row[f"{label} time s"] = f"{e.runtime.mean():.1f}" if len(e) else "-"
        row.update({"CW gap %": round(h.cw_gap_pct.mean(), 2), "LS gap %": round(h.ls_gap_pct.mean(), 2),
                    "SA gap %": round(h.sa_gap_pct.mean(), 2), "SA time s": f"{h.sa_time_s.mean():.2f}",
                    "SA feasible": f"{int(h.sa_feasible.sum())}/{len(h)}"})
        rows.append(row)
    _save_table(pd.DataFrame(rows), "T5_tiers")


def scaling_table(sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, g in sc.groupby("n"):
        h = g[g.method == SA_METHOD]
        opt = g[g.status == "optimal"].cost
        reference = opt.min() if not opt.empty else g.cost.min()
        row = {"n": n, "Reference cost": int(reference), "Proven optimal": not opt.empty}
        for model in ("basic", "strengthened"):
            e = g[g.method == f"MILP ({model})"].iloc[0]
            row[f"MILP {model} status"] = _status(e)
            row[f"MILP {model} cost"] = None if pd.isna(e.cost) else int(e.cost)
            row[f"MILP {model} time s"] = f"{e.runtime:.1f}"
        row["SA mean cost"] = f"{h.cost.mean():.1f}"
        row["SA runs at reference"] = f"{int((h.cost == reference).sum())}/{len(h)}"
        row["SA mean time s"] = f"{h.runtime.mean():.2f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    for model in ("basic", "strengthened"):
        df[f"MILP {model} cost"] = df[f"MILP {model} cost"].astype("Int64")
    _save_table(df, "T6_scaling")
    return df


# ----- figures -----------------------------------------------------------------------------
def fig_exact_bounds(ex: pd.DataFrame) -> None:
    order = ex.drop_duplicates("instance").sort_values("n").instance.tolist()
    y = np.arange(len(order))[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for i, model in enumerate(("basic", "strengthened")):
        e = ex[ex.model == model].set_index("instance").loc[order]
        c = METHOD_COLOURS[f"MILP ({model})"]
        offset = 0.18 if i == 0 else -0.18
        ax1.barh(y + offset, e.root_gap, height=0.34, color=c, label=f"MILP ({model})")
        ax2.barh(y + offset, e.gap.fillna(100.0), height=0.34, color=c, label=f"MILP ({model})")
        for yi, status in zip(y, e.status):
            if status == "optimal":
                ax2.text(0.5, yi + offset, "proven optimal", va="center", fontsize=7, color=INK_2)
    ax1.set_yticks(y, order)
    ax1.set_xlabel("LP relaxation gap to optimum (%)")
    ax1.set_title("How weak is the LP relaxation? (root gap)", loc="left")
    ax2.set_xlabel("optimality gap left after the time limit (%)")
    ax2.set_title("What branch-and-bound could prove", loc="left")
    ax1.legend(loc="lower right")
    fig.tight_layout()
    _save_fig(fig, "F4_exact_bounds.png")


def fig_scaling(sc: pd.DataFrame, table: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for method in ("MILP (basic)", "MILP (strengthened)"):
        g = sc[sc.method == method].sort_values("n")
        c = METHOD_COLOURS[method]
        ax1.plot(g.n, g.runtime, color=c, label=method)
        proven = g.status == "optimal"
        ax1.plot(g.n[proven], g.runtime[proven], "o", color=c, ms=6)
        ax1.plot(g.n[~proven], g.runtime[~proven], "o", mfc="white", mec=c, mew=1.5, ms=6)
    h = sc[sc.method == SA_METHOD].groupby("n").runtime.agg(["mean", "min", "max"])
    c = METHOD_COLOURS[SA_METHOD]
    ax1.plot(h.index, h["mean"], color=c, marker="D", ms=5, label=f"{SA_METHOD}, mean of 10 seeds")
    ax1.fill_between(h.index, h["min"], h["max"], color=c, alpha=0.12, lw=0)
    limit = sc[sc.method.str.startswith("MILP")].runtime.max()
    ax1.axhline(limit, color=MUTED, lw=1)
    ax1.text(sc.n.min(), limit * 1.15, "time limit (hollow = stopped, no proof)", color=INK_2, fontsize=8)
    ax1.set_yscale("log")
    ax1.set_ylim(top=limit * 3)
    ax1.set_xlabel(f"customers n (first n customers of {SCALING_BASE})")
    ax1.set_ylabel("runtime (s, log scale)")
    ax1.set_title("Runtime: exponential for the MILP, flat for SA", loc="left")
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))

    for method in ("MILP (basic)", "MILP (strengthened)"):
        col = f"MILP {method[6:-1]} cost"
        gap = 100 * (table[col] - table["Reference cost"]) / table["Reference cost"]
        ax2.plot(table.n, gap, marker="o", color=METHOD_COLOURS[method], label=method)
    gap = 100 * (table["SA mean cost"].astype(float) - table["Reference cost"]) / table["Reference cost"]
    ax2.plot(table.n, gap, marker="D", ms=5, color=c, label=f"{SA_METHOD}, mean")
    ax1.set_xticks(table.n)
    ax2.set_xticks(table.n)
    ax2.set_xlabel("customers n")
    ax2.set_ylabel("gap to best cost found (%)")
    ax2.set_title("Quality at the same sizes", loc="left")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    _save_fig(fig, "scaling_exact_vs_sa.png")


def fig_heuristic_tiers(he: pd.DataFrame) -> None:
    """Mean gap of each heuristic layer per tier, computed from results/heuristic_results.csv."""
    means = he.groupby("tier")[["cw_gap_pct", "ls_gap_pct", "sa_gap_pct"]].mean().loc[TIER_ORDER]
    labels = ["Clarke-Wright savings", "+ local search (VND)", "Simulated annealing"]
    x = np.arange(len(TIER_ORDER))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for k, (col, label) in enumerate(zip(means.columns, labels)):
        bars = ax.bar(x + (k - 1) * width, means[col], width - 0.02, color=ORDINAL_BLUES[k], label=label)
        for bar, value in zip(bars, means[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{value:.2f}", ha="center", fontsize=7, color=INK_2)
    counts = he.tier.value_counts()
    ax.set_xticks(x, [f"{t} ({counts[t]} instances)" for t in TIER_ORDER])
    ax.set_ylabel("mean gap to optimum (%)")
    ax.set_title("Mean optimality gap of each heuristic layer, per tier", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_fig(fig, "heuristic_tier_gaps.png")


def main() -> None:
    style()
    ex = pd.read_csv(RESULTS / "exact.csv")
    sc = pd.read_csv(RESULTS / "scaling.csv")
    he = pd.read_csv(RESULTS / "heuristic_results.csv")
    instances_table()
    exact_table(ex)
    exact_compact_table(ex)
    heuristic_table(he)
    tier_table(ex, he)
    fig_scaling(sc, scaling_table(sc))
    fig_exact_bounds(ex)
    fig_heuristic_tiers(he)


if __name__ == "__main__":
    main()

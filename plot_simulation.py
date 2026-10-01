"""Route maps for the Bengaluru MDC scenarios - one panel per case.

usage: python plot_simulation.py [results.xlsx] [constraints.xlsx]
       (defaults: Bengaluru_MDC_Simulation_CostOptimized.xlsx, Banglore Constraint.xlsx;
        PNG written alongside the results)
"""
import math
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BOOK = sys.argv[1] if len(sys.argv) > 1 else \
    r"D:\PHD Clustering\Bengaluru_MDC_Simulation_CostOptimized.xlsx"
CUST = sys.argv[2] if len(sys.argv) > 2 else r"D:\PHD Clustering\Banglore Constraint.xlsx"

allc = pd.read_excel(CUST, sheet_name="CustomerBase")
xl = pd.ExcelFile(BOOK)
cases = pd.read_excel(xl, sheet_name="Case_Comparison").Case.tolist()
locs = pd.read_excel(xl, sheet_name="MDC_Locations")
if "Site" not in locs.columns:
    locs["Site"] = locs.MDC

ncol = 2
nrow = math.ceil(len(cases) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(11 * ncol, 11 * nrow), squeeze=False)
axes = axes.ravel()

for ax, case in zip(axes, cases):
    routes = pd.read_excel(xl, sheet_name=f"{case}_Routes")
    summ = pd.read_excel(xl, sheet_name=f"{case}_VehicleSummary")
    cl = locs[locs.Case == case]
    mdc = {r.MDC: (r.Latitude, r.Longitude, r.Site) for r in cl.itertuples()}

    ax.scatter(allc.Longitude, allc.Latitude, s=26, c="#d9d9d9",
               edgecolor="#9a9a9a", linewidth=.5, zorder=1)

    cmap = plt.get_cmap("tab20")
    for i, (veh, g) in enumerate(routes.groupby("Vehicle", sort=False)):
        g = g.sort_values("StopNumber")
        home = mdc[g.MDC.iloc[0]]
        xs = [home[1]] + g.Longitude.tolist()
        ys = [home[0]] + g.Latitude.tolist()
        is_ev = g.VehicleType.iloc[0] != "Bike"      # EV / Auto drawn thick
        ax.plot(xs, ys, "-", color=cmap(i % 20), zorder=2,
                linewidth=2.6 if is_ev else 1.3,
                alpha=.95 if is_ev else .75)
        ax.scatter(g.Longitude, g.Latitude, s=34, color=cmap(i % 20),
                   edgecolor="white", linewidth=.6, zorder=3)

    served = set(routes.CustomerId)
    miss = allc[~allc.CustomerId.isin(served)]
    if len(miss):
        ax.scatter(miss.Longitude, miss.Latitude, s=210, marker="X",
                   c="#d62728", edgecolor="black", linewidth=1.2, zorder=6)
        for _, r in miss.iterrows():
            ax.annotate("UNSERVED", (r.Longitude, r.Latitude),
                        textcoords="offset points", xytext=(9, -4),
                        fontsize=9, weight="bold", color="#d62728")

    for name, (la, lo, site) in mdc.items():
        ax.scatter([lo], [la], s=460, marker="*", c="#111111", zorder=7)
        label = name if site == name else f"{name}\n({site})"
        ax.annotate(label, (lo, la), textcoords="offset points", xytext=(11, 8),
                    fontsize=12, weight="bold")

    mix = ", ".join(f"{n} {t}" for t, n in summ.VehicleType.value_counts().sort_index().items())
    cost = (f"  |  Rs {summ.TotalCost.sum():,.0f}"
            if "TotalCost" in summ.columns else "")
    sites = " + ".join(n if s == n else f"{n} {s}" for n, (_, _, s) in mdc.items())
    ax.set_title(
        f"{case} - {sites}\n"
        f"{len(summ)} vehicles ({mix})  |  "
        f"{int(summ.Stops.sum())}/{len(allc)} customers  |  "
        f"one-way {summ.OneWayDistance_km.sum():.0f} km{cost}",
        fontsize=14, weight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect("equal"); ax.grid(alpha=.15)

for ax in axes[len(cases):]:
    ax.axis("off")

axes[0].legend(handles=[
    Line2D([], [], color="#555", lw=2.8, label="EV / Auto route"),
    Line2D([], [], color="#555", lw=1.3, label="Bike route"),
    Line2D([], [], marker="*", color="w", markerfacecolor="#111", markersize=17, label="MDC"),
    Line2D([], [], marker="X", color="w", markerfacecolor="#d62728", markersize=12, label="Unserved"),
], loc="lower left", fontsize=10, framealpha=.95)

lo0 = min(allc.Longitude.min(), locs.Longitude.min()) - .02
lo1 = max(allc.Longitude.max(), locs.Longitude.max()) + .02
la0 = min(allc.Latitude.min(), locs.Latitude.min()) - .02
la1 = max(allc.Latitude.max(), locs.Latitude.max()) + .02
for ax in axes[:len(cases)]:
    ax.set_xlim(lo0, lo1); ax.set_ylim(la0, la1)

objective = "distance-optimised" if os.path.basename(BOOK) == "Bengaluru_MDC_Simulation.xlsx" \
    else "cost-optimised"
city = os.path.basename(BOOK).split("_")[0]
fig.suptitle(f"{city} MDC Simulation ({objective}) - one-way routing (MDC to last customer), "
             f"OSRM road distances", fontsize=17, weight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.splitext(BOOK)[0] + ".png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)

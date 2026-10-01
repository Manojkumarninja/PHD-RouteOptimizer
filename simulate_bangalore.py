"""
simulate_bangalore.py - Simulate the MDC scenarios in a Bengaluru constraint workbook
using the Route Optimizer engine.

usage: python simulate_bangalore.py [constraints.xlsx] [output.xlsx]

Cases, MDCs and the fleet are read from the 'Constraints' sheet.

Heterogeneous fleet: each vehicle carries its own limits on
  * drop points   (max stops on the route)
  * distance      (ONE-WAY km, MDC -> ... -> last customer; no return leg)
  * tonnage       (total load)
and its own cost: Fixed Cost (per vehicle used) + VariableCost/km x one-way km.

Objective: minimise total Rs cost while serving every customer.
Distances are real road distances from OSRM, Haversine fallback.
"""

import os
import re
import sys
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizer import haversine_km, fetch_osrm_matrix   # reuse the app's helpers

BOOK = sys.argv[1] if len(sys.argv) > 1 else r"D:\PHD Clustering\Banglore Constraint.xlsx"
OUTNAME = sys.argv[2] if len(sys.argv) > 2 else "Bengaluru_MDC_Simulation_CostOptimized.xlsx"
BASELINE = os.environ.get("BASELINE", "")   # earlier results workbook to re-price, optional
OUTDIR = r"D:\PHD Clustering"

SPEED_KMH = 20.0          # app default
BUFFER_MIN = 3.0          # app default - per customer
WAITING_MIN = 5.0         # app default - per customer
SOLVER_SECONDS = int(os.environ.get('SOLVER_SECONDS', 300))


# ──────────────── read the fleet from the Constraints sheet ────────────────

def read_constraints(book: str = BOOK) -> Dict[str, Dict]:
    """
    Parse the 'Constraints' sheet into {case: {depots: [...], fleet: [...]}}.

    Layout expected (blocks may sit side by side):
      * a facility table   : Case | Facility | Latitude | Longitude
      * one vehicle table per case+MDC, titled e.g. 'Case1-MDC1' / 'Case2 - MDC2',
        with header row      Vehicle | Drop Points | Distance | Tonnage
    """
    raw = pd.read_excel(book, sheet_name="Constraints", header=None)
    nrow, ncol = raw.shape

    def cell(r, c):
        if 0 <= r < nrow and 0 <= c < ncol:
            v = raw.iat[r, c]
            return "" if pd.isna(v) else str(v).strip()
        return ""

    def case_key(txt):
        m = re.search(r"case\s*(\d+)", txt, re.I)
        return f"Case{m.group(1)}" if m else None

    # ── facility coordinates, per case (the same MDC code can sit at a
    #    different site in each case, e.g. 'MDC2 ( Munishwara)') ────────────
    coords: Dict[Tuple[str, str], Tuple[float, float]] = {}
    sites: Dict[Tuple[str, str], str] = {}
    for r in range(nrow):
        for c in range(ncol):
            if cell(r, c).lower() == "case" and cell(r, c + 1).lower() == "facility":
                cur, blanks = None, 0
                for rr in range(r + 1, nrow):
                    name = cell(rr, c + 1)
                    if not name:
                        blanks += 1
                        if blanks >= 3 and cur:       # table has ended
                            break
                        continue
                    blanks = 0
                    cur = case_key(cell(rr, c)) or cur
                    try:
                        la, lo = float(raw.iat[rr, c + 2]), float(raw.iat[rr, c + 3])
                    except (TypeError, ValueError):
                        continue
                    m = re.match(r"(MDC\s*\d+)", name, re.I)
                    if not m or math.isnan(la) or not cur:
                        continue                      # sub-heading row like 'Case 2'
                    code = m.group(1).replace(" ", "").upper()
                    coords[(cur, code)] = (la, lo)
                    site = re.search(r"\(\s*(.*?)\s*\)", name)
                    sites[(cur, code)] = site.group(1) if site else code
                break

    # ── vehicle tables ('Case1-MDC1', 'Case2,3,4 - MDC2', ...) ─────────
    cases: Dict[str, Dict] = {}
    for r in range(nrow):
        for c in range(ncol):
            if (cell(r, c).lower() == "vehicle"
                    and cell(r, c + 1).lower().startswith("drop")
                    and cell(r, c + 2).lower().startswith("dist")):
                title = ""
                for back in (1, 2, 3):
                    if cell(r - back, c):
                        title = cell(r - back, c)
                        break
                m = re.match(r"Case\s*([\d,\s&]+?)\s*-\s*(MDC\s*\d+)", title, re.I)
                if not m:
                    continue
                case_list = [f"Case{n}" for n in re.findall(r"\d+", m.group(1))]
                mdc = m.group(2).replace(" ", "").upper()

                # locate this block's columns by header name
                hdr = {}
                for cc in range(c, ncol):
                    h = re.sub(r"[^a-z]", "", cell(r, cc).lower())
                    if not h:
                        break
                    hdr[h] = cc
                col_fixed = next((cc for h, cc in hdr.items() if h.startswith("fixed")), None)
                col_var = next((cc for h, cc in hdr.items() if h.startswith("variable")), None)

                def num(rr, cc):
                    if cc is None or pd.isna(raw.iat[rr, cc]):
                        return 0.0
                    return float(raw.iat[rr, cc])

                rows = []
                for rr in range(r + 1, nrow):
                    v = cell(rr, c)
                    if not v:
                        if rows:
                            break
                        continue
                    drops = num(rr, c + 1)
                    if drops != int(drops):
                        # fractional drop points (e.g. 3.5) = fleet average:
                        # alternate vehicles between the floor and the ceiling
                        drops = math.ceil(drops) if len(rows) % 2 == 0 else math.floor(drops)
                    rows.append((v, int(drops), num(rr, c + 2), num(rr, c + 3),
                                 num(rr, col_fixed), num(rr, col_var)))

                # group identical vehicles into
                # (label, count, drops, dist, ton, home, fixed_cost, var_cost_per_km)
                buckets: Dict[Tuple, int] = {}
                for name, drops, dkm, ton, fixed, var in rows:
                    label = re.sub(r"\s*\d+$", "", name).strip() or "Veh"
                    key = (label, drops, dkm, ton, fixed, var)
                    buckets[key] = buckets.get(key, 0) + 1
                for case in case_list:
                    cases.setdefault(case, {"depots": [], "fleet": [], "sites": {}, "vehicles": []})
                    if mdc not in [d[0] for d in cases[case]["depots"]]:
                        cases[case]["depots"].append((mdc, *coords[(case, mdc)]))
                        cases[case]["sites"][mdc] = sites[(case, mdc)]
                    for (label, drops, dkm, ton, fixed, var), n in buckets.items():
                        cases[case]["fleet"].append((label, n, drops, dkm, ton, mdc, fixed, var))
                    # the same fleet, one entry per vehicle with its own sheet name
                    for name, drops, dkm, ton, fixed, var in rows:
                        cases[case]["vehicles"].append({
                            "MDC": mdc, "Vehicle": name, "DropPoints": drops,
                            "MaxDistance_km": dkm, "MaxTonnage_kg": ton,
                            "FixedCost_Rs": fixed, "VariableCost_Rs_per_km": var})

    for case in cases:
        cases[case]["depots"].sort(key=lambda d: d[0])
    return dict(sorted(cases.items(), key=lambda kv: int(kv[0][4:])))


# filled by main() - nothing is read at import time, so the Streamlit app
# can import this module on a machine without the local workbooks
CASES: Dict[str, Dict] = {}


# ─────────────────────────── helpers ───────────────────────────

def parse_window(s: str) -> Tuple[int, int]:
    """'9:30-13:30' -> (570, 810) minutes from midnight."""
    m = re.findall(r"(\d{1,2}):(\d{2})", str(s))
    if len(m) != 2:
        return (0, 1440)
    return tuple(int(h) * 60 + int(mi) for h, mi in m)


def hhmm(minutes: float) -> str:
    h, m = divmod(int(round(minutes)) % 1440, 60)
    return f"{h:02d}:{m:02d}"


def shift_text(start, end) -> str:
    if start is None and end is None:
        return ""
    if end is None:
        return f"from {hhmm(start)}"
    if start is None:
        return f"until {hhmm(end)}"
    return f"{hhmm(start)}-{hhmm(end)}"


def build_matrix(lats: List[float], lons: List[float]) -> Tuple[np.ndarray, str]:
    try:
        dist, _ = fetch_osrm_matrix(lats, lons)
        return dist, "OSRM (real road)"
    except Exception as exc:
        print(f"  ! OSRM unavailable ({exc}); using Haversine")
        n = len(lats)
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    d[i][j] = haversine_km(lats[i], lons[i], lats[j], lons[j])
        return d, "Haversine (straight line)"


# ─────────────────────────── solver ────────────────────────────

def solve_case(case_name: str, cfg: Dict, cust: pd.DataFrame,
               vehicle_penalty_rs: float = 0.0) -> Dict:
    """
    vehicle_penalty_rs: an extra Rs charged to the OBJECTIVE (not to the
        reported cost) for every vehicle dispatched. 0 = pure cost
        minimisation. A large value (e.g. 10,000) makes the solver use the
        fewest vehicles first and only then minimise cost - otherwise vehicles
        with zero fixed cost are free to dispatch, and since the return leg is
        free too, customers on opposite sides of an MDC get split across bikes.
    cfg["fleet"] tuples may carry optional extra fields: [8], [9] the
        vehicle's shift start / end in minutes from midnight (None = no
        limit); [10] an explicit vehicle name (used when count == 1), so a
        per-vehicle fleet keeps its sheet names, e.g. "MDC2-Bike7".
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    depots = cfg["depots"]
    n_dep = len(depots)
    n_cus = len(cust)
    n_nodes = n_dep + n_cus
    depot_idx = {name: i for i, (name, _, _) in enumerate(depots)}

    lats = [d[1] for d in depots] + cust["Latitude"].astype(float).tolist()
    lons = [d[2] for d in depots] + cust["Longitude"].astype(float).tolist()

    dist, source = build_matrix(lats, lons)
    time_mat = np.round(dist / SPEED_KMH * 60).astype(int)

    # expand the fleet into individual vehicles
    veh_name, veh_drops, veh_dist, veh_ton, veh_start = [], [], [], [], []
    veh_fixed, veh_var, veh_type, veh_shift = [], [], [], []
    seen: Dict[Tuple[str, str], int] = {}
    for f in cfg["fleet"]:
        label, count, drops, dkm, ton, home, fixed, var = f[:8]
        shift = (f[8], f[9]) if len(f) >= 10 else (None, None)
        explicit = f[10] if len(f) >= 11 and f[10] and count == 1 else None
        for _ in range(count):
            veh_shift.append(shift)
            seen[(home, label)] = seen.get((home, label), 0) + 1
            veh_name.append(f"{home}-{explicit}" if explicit
                            else f"{home}-{label}{seen[(home, label)]}")
            veh_type.append(label)
            veh_drops.append(drops)
            veh_dist.append(int(round(dkm * 1000)))     # metres
            veh_ton.append(int(round(ton * 1000)))      # grams (demands x1000)
            veh_start.append(depot_idx[home])
            veh_fixed.append(fixed)                     # Rs per vehicle used
            veh_var.append(var)                         # Rs per km
    n_veh = len(veh_name)

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_veh, veh_start, veh_start)
    routing = pywrapcp.RoutingModel(manager)

    dist_m = np.round(dist * 1000).astype(int)
    is_depot = [i < n_dep for i in range(n_nodes)]

    # ONE-WAY: the leg back into a depot is free and is not counted
    def one_way_dist(fi, ti):
        a, b = manager.IndexToNode(fi), manager.IndexToNode(ti)
        return 0 if is_depot[b] else int(dist_m[a][b])

    dist_cb = routing.RegisterTransitCallback(one_way_dist)

    # per-vehicle ONE-WAY distance cap (MDC -> ... -> last customer)
    routing.AddDimensionWithVehicleCapacity(
        dist_cb, 0, veh_dist, True, "Distance")

    # ── objective = Rs cost, in paise so everything stays integer ────
    #   variable: Rs/km x one-way km   -> paise per metre = var / 10
    #   fixed   : charged once for every vehicle that leaves its MDC
    var_cbs = {}
    for v in range(n_veh):
        rate = veh_var[v]
        if rate not in var_cbs:
            def _mk(rate_=rate):
                def _cb(fi, ti):
                    return int(round(one_way_dist(fi, ti) * rate_ / 10))
                return _cb
            var_cbs[rate] = routing.RegisterTransitCallback(_mk())
        routing.SetArcCostEvaluatorOfVehicle(var_cbs[rate], v)
        routing.SetFixedCostOfVehicle(
            int(round((veh_fixed[v] + vehicle_penalty_rs) * 100)), v)

    # service + travel time, with the delivery window
    svc = [0] * n_dep + [int(round(BUFFER_MIN + WAITING_MIN))] * n_cus

    # ONE-WAY here too: the drive back to the MDC must not count against the
    # delivery window, or long routes get split just to be "home" in time
    def time_cb(fi, ti):
        a, b = manager.IndexToNode(fi), manager.IndexToNode(ti)
        return (0 if is_depot[b] else int(time_mat[a][b])) + svc[a]

    time_idx = routing.RegisterTransitCallback(time_cb)
    win = [parse_window(w) for w in cust["DeliveryWindow"]]
    day_open = min(s for s, _ in win)
    day_close = max(e for _, e in win)
    horizon = min(1440, max([day_close] + [e for _, e in veh_shift if e is not None]))
    routing.AddDimension(time_idx, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for c in range(n_cus):
        s, e = win[c]
        time_dim.CumulVar(manager.NodeToIndex(n_dep + c)).SetRange(s, e)
    for v in range(n_veh):
        s, e = veh_shift[v]
        start_lo = day_open if s is None else s
        end_hi = horizon if e is None else min(e, horizon)
        time_dim.CumulVar(routing.Start(v)).SetRange(min(start_lo, end_hi), end_hi)
        time_dim.CumulVar(routing.End(v)).SetMax(end_hi)

    # tonnage
    dem = [0] * n_dep + [int(round(float(t) * 1000)) for t in cust["Tonnage"]]

    def dem_cb(fi):
        return dem[manager.IndexToNode(fi)]

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(dem_cb), 0, veh_ton, True, "Tonnage")

    # drop points
    def stop_cb(fi):
        return 0 if is_depot[manager.IndexToNode(fi)] else 1

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(stop_cb), 0, veh_drops, True, "Stops")

    # every customer must be served if at all possible
    # (penalty = Rs 1 crore in paise - far above any real plan cost)
    for node in range(n_dep, n_nodes):
        routing.AddDisjunction([manager.NodeToIndex(node)], 1_000_000_000)

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = SOLVER_SECONDS
    sol = routing.SolveWithParameters(p)
    if sol is None:
        raise RuntimeError(f"{case_name}: no feasible solution")

    # ── read the solution ────────────────────────────────────────
    stop_rows, summ_rows = [], []
    served = set()
    for v in range(n_veh):
        idx = routing.Start(v)
        if routing.IsEnd(sol.Value(routing.NextVar(idx))):
            continue                                   # vehicle unused

        home = depots[veh_start[v]][0]
        seq, one_way_km, prev, clock = [], 0.0, veh_start[v], None
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node >= n_dep:
                c = node - n_dep
                leg = float(dist[prev][node])
                one_way_km += leg
                arrive = sol.Value(time_dim.CumulVar(idx))
                seq.append((c, leg, arrive))
                served.add(c)
                prev = node
            idx = sol.Value(routing.NextVar(idx))

        load = sum(float(cust.iloc[c]["Tonnage"]) for c, _, _ in seq)
        for n, (c, leg, arrive) in enumerate(seq, 1):
            r = cust.iloc[c]
            stop_rows.append({
                "Case": case_name, "MDC": home, "Vehicle": veh_name[v],
                "VehicleType": veh_type[v],
                "StopNumber": n, "TotalStops": len(seq),
                "CustomerId": r["CustomerId"], "Customer": r["Customer"],
                "Latitude": r["Latitude"], "Longitude": r["Longitude"],
                "Tonnage": r["Tonnage"], "DeliveryWindow": r["DeliveryWindow"],
                "LegDistance_km": round(leg, 2),
                "CumulativeDistance_km": round(sum(s[1] for s in seq[:n]), 2),
                "EstimatedArrival": hhmm(arrive),
            })
        summ_rows.append({
            "Case": case_name, "MDC": home, "Vehicle": veh_name[v],
            "VehicleType": veh_type[v],
            "Stops": len(seq), "MaxStops": veh_drops[v],
            "Tonnage": round(load, 3), "MaxTonnage": veh_ton[v] / 1000,
            "OneWayDistance_km": round(one_way_km, 2),
            "MaxDistance_km": veh_dist[v] / 1000,
            "FixedCost": veh_fixed[v],
            "VariableCostPerKm": veh_var[v],
            "VariableCost": round(one_way_km * veh_var[v], 2),
            "TotalCost": round(veh_fixed[v] + one_way_km * veh_var[v], 2),
            "CostPerDrop": round((veh_fixed[v] + one_way_km * veh_var[v]) / len(seq), 2),
            "LastCustomer": cust.iloc[seq[-1][0]]["Customer"],
            "FirstArrival": hhmm(seq[0][2]), "LastArrival": hhmm(seq[-1][2]),
            "Shift": shift_text(*veh_shift[v]),
        })

    return {
        "case": case_name,
        "stops": pd.DataFrame(stop_rows),
        "summary": pd.DataFrame(summ_rows),
        "unserved": sorted(set(range(n_cus)) - served),
        "source": source,
        "fleet_size": n_veh,
    }


# ──────────────────────────── main ─────────────────────────────

def main():
    global CASES
    CASES = read_constraints(BOOK)
    cust = pd.read_excel(BOOK, sheet_name="CustomerBase").reset_index(drop=True)
    print(f"Customers: {len(cust)}   Total tonnage: {cust.Tonnage.sum():g}")

    results = {}
    for name, cfg in CASES.items():
        print(f"\n=== {name} : {[d[0] for d in cfg['depots']]} ===")
        res = solve_case(name, cfg, cust)
        results[name] = res
        s = res["summary"]
        print(f"  distances: {res['source']}")
        print(f"  vehicles used {len(s)} of {res['fleet_size']}  |  "
              f"stops {int(s.Stops.sum())}/{len(cust)}  |  "
              f"one-way km total {s.OneWayDistance_km.sum():.2f}  |  "
              f"COST Rs {s.TotalCost.sum():,.2f} "
              f"(fixed {s.FixedCost.sum():,.0f} + variable {s.VariableCost.sum():,.2f})")
        for t, g in s.groupby("VehicleType"):
            print(f"    {t:5s} x{len(g):2d}  stops {int(g.Stops.sum()):3d}  "
                  f"tonnage {g.Tonnage.sum():8.2f}  "
                  f"one-way km {g.OneWayDistance_km.sum():7.2f} "
                  f"(avg {g.OneWayDistance_km.mean():.2f}, max {g.OneWayDistance_km.max():.2f})")
        if res["unserved"]:
            print(f"  !! UNSERVED {len(res['unserved'])}: "
                  f"{[cust.iloc[i]['Customer'] for i in res['unserved']]}")

    # baseline: the earlier distance-optimised plan, re-priced at today's rates
    baseline = {}
    prev = BASELINE
    if prev and os.path.exists(prev):
        px = pd.ExcelFile(prev)
        for name, cfg in CASES.items():
            sheet = f"{name}_VehicleSummary"
            if sheet not in px.sheet_names:
                continue
            ps = pd.read_excel(px, sheet_name=sheet)
            rate = {(f[5], f[0]): (f[6], f[7]) for f in cfg["fleet"]}
            fixed = sum(rate[(r.MDC, r.VehicleType)][0] for r in ps.itertuples())
            var = sum(rate[(r.MDC, r.VehicleType)][1] * r.OneWayDistance_km for r in ps.itertuples())
            baseline[name] = {"vehicles": len(ps), "km": ps.OneWayDistance_km.sum(),
                              "served": int(ps.Stops.sum()),
                              "fixed": fixed, "var": var, "cost": fixed + var}

    out = os.path.join(OUTDIR, OUTNAME)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        comp = []
        for name, res in results.items():
            s = res["summary"]
            b = baseline.get(name)
            comp.append({
                "Case": name,
                "MDCs": ", ".join(f"{d[0]} ({CASES[name]['sites'][d[0]]})"
                                  if CASES[name]["sites"][d[0]] != d[0] else d[0]
                                  for d in CASES[name]["depots"]),
                "VehiclesAvailable": res["fleet_size"],
                "VehiclesUsed": len(s),
                **{f"{t}Used": int((s.VehicleType == t).sum())
                   for t in dict.fromkeys(f[0] for f in CASES[name]["fleet"])},
                "CustomersServed": int(s.Stops.sum()),
                "CustomersUnserved": len(res["unserved"]),
                "TotalTonnage": round(s.Tonnage.sum(), 2),
                "TotalOneWayDistance_km": round(s.OneWayDistance_km.sum(), 2),
                "AvgOneWayDistance_km": round(s.OneWayDistance_km.mean(), 2),
                "MaxOneWayDistance_km": round(s.OneWayDistance_km.max(), 2),
                "FixedCost": round(s.FixedCost.sum(), 2),
                "VariableCost": round(s.VariableCost.sum(), 2),
                "TotalCost": round(s.TotalCost.sum(), 2),
                "CostPerCustomer": round(s.TotalCost.sum() / max(int(s.Stops.sum()), 1), 2),
                "CostPerKg": round(s.TotalCost.sum() / max(s.Tonnage.sum(), 1e-9), 2),
                "Baseline_DistOpt_Vehicles": b["vehicles"] if b else None,
                "Baseline_DistOpt_km": round(b["km"], 2) if b else None,
                "Baseline_DistOpt_Cost": round(b["cost"], 2) if b else None,
                "Saving_vs_DistOpt": round(b["cost"] - s.TotalCost.sum(), 2) if b else None,
                "DistanceSource": res["source"],
            })
        pd.DataFrame(comp).to_excel(xl, sheet_name="Case_Comparison", index=False)
        if baseline:
            print("\nBaseline (earlier distance-optimised plan, priced at these rates):")
            for n, b in baseline.items():
                print(f"  {n}: {b['vehicles']} vehicles, {b['km']:.2f} km, "
                      f"{b['served']} served -> Rs {b['cost']:,.2f}")
        for name, res in results.items():
            res["summary"].to_excel(xl, sheet_name=f"{name}_VehicleSummary", index=False)
            res["stops"].to_excel(xl, sheet_name=f"{name}_Routes", index=False)
        pd.DataFrame([
            {"Case": n, "MDC": d[0], "Site": c["sites"][d[0]], "Latitude": d[1], "Longitude": d[2]}
            for n, c in CASES.items() for d in c["depots"]
        ]).to_excel(xl, sheet_name="MDC_Locations", index=False)
    print(f"\nWrote {out}")
    return results


if __name__ == "__main__":
    main()

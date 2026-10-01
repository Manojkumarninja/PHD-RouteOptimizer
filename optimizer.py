"""
optimizer.py — VRP solver with time windows and capacity constraints.
Supports Google OR-Tools (optimal) and greedy nearest-neighbour (fast fallback).
Distance: OSRM (real roads, free) | OpenRouteService (real roads, API key) | Haversine (offline).
Features: Slot-mixed or same-slot-only trips | Round-trip or one-way routes | Crate capacity | Service time.
"""

import math
import re
import requests
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Optional, Any


# ──────────────────────────── Utilities ────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(Δλ / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def parse_slot_minutes(slot_str: str):
    """Parse '5:00 AM - 6:00 AM' → (start_min, end_min) from midnight."""
    try:
        parts = re.split(r"\s*-\s*", str(slot_str).strip(), maxsplit=1)
        if len(parts) != 2:
            return (0, 1440)

        def _to_min(s: str) -> int:
            s = s.strip()
            m = re.match(r"(\d+)(?::(\d+))?\s*(AM|PM)", s, re.IGNORECASE)
            if not m:
                return 0
            h, mn = int(m.group(1)), int(m.group(2) or 0)
            period = m.group(3).upper()
            if period == "PM" and h != 12:
                h += 12
            if period == "AM" and h == 12:
                h = 0
            return h * 60 + mn

        return (_to_min(parts[0]), _to_min(parts[1]))
    except Exception:
        return (0, 1440)


def minutes_to_hhmm(minutes: int) -> str:
    h, m = divmod(int(minutes) % 1440, 60)
    period = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12:02d}:{m:02d} {period}"


def group_unique_slots(df: pd.DataFrame) -> List[List[int]]:
    """
    Partition customer indices into groups so that within each group every
    slot string is unique — i.e. no two customers in the same group share the
    same delivery slot.
    """
    from collections import defaultdict

    slot_buckets: Dict[str, List[int]] = defaultdict(list)
    for ci, (_, row) in enumerate(df.iterrows()):
        slot_buckets[str(row["Slot"]).strip()].append(ci)

    n_groups = max(len(v) for v in slot_buckets.values())
    groups: List[List[int]] = [[] for _ in range(n_groups)]

    for customers_in_slot in slot_buckets.values():
        for i, ci in enumerate(customers_in_slot):
            groups[i].append(ci)

    return [g for g in groups if g]


# ──────────────────── Real-Road Matrix Fetchers ────────────────────

def fetch_osrm_matrix(lats: List[float], lons: List[float]) -> tuple:
    """Real road distances & times from OSRM public server (free, no key)."""
    coords = ";".join(f"{lon},{lat}" for lat, lon in zip(lats, lons))
    url = (
        f"http://router.project-osrm.org/table/v1/driving/{coords}"
        "?annotations=duration,distance"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data.get('message', 'unknown')}")

    n = len(lats)
    raw_dur = data.get("durations", [])
    raw_dis = data.get("distances", [])
    time_mat = np.zeros((n, n))
    dist_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dur = raw_dur[i][j] if raw_dur and raw_dur[i][j] is not None else None
            dis = raw_dis[i][j] if raw_dis and raw_dis[i][j] is not None else None
            time_mat[i][j] = dur / 60 if dur is not None else (
                haversine_km(lats[i], lons[i], lats[j], lons[j]) / 20 * 60
            )
            dist_mat[i][j] = dis / 1000 if dis is not None else (
                haversine_km(lats[i], lons[i], lats[j], lons[j])
            )

    return dist_mat, np.round(time_mat).astype(int)


def fetch_ors_matrix(lats: List[float], lons: List[float], api_key: str) -> tuple:
    """Real road distances & times from OpenRouteService (free API key required)."""
    locations = [[lon, lat] for lat, lon in zip(lats, lons)]
    resp = requests.post(
        "https://api.openrouteservice.org/v2/matrix/driving-hgv",
        json={"locations": locations, "metrics": ["duration", "distance"], "units": "km"},
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    durations = np.array(data["durations"])
    distances = np.array(data["distances"])
    return distances, np.round(durations / 60).astype(int)


# ──────────────────────── Problem Builder ──────────────────────────

def build_problem_data(
    df: pd.DataFrame,
    num_vehicles: int,
    max_drops: int,
    max_tonnage: float,
    speed_kmh: float,
    routing: str = "haversine",
    ors_api_key: str = "",
    round_trip: bool = True,
    unique_slots: bool = False,
    max_crates: int = 0,
    loading_sec_per_crate: float = 30.0,
    unloading_sec_per_crate: float = 45.0,
    buffer_time_min: float = 3.0,
    waiting_time_min: float = 5.0,
    max_service_time_min: float = 45.0,
) -> Dict[str, Any]:
    """
    Build the VRP data dictionary from orders dataframe.
    round_trip:   True  → vehicle returns to FC; distance includes return leg.
                  False → one-way; only outward distance counted.
    unique_slots: True  → each delivery slot may appear at most ONCE per route.
    max_crates:   0     → crate constraint disabled.
                  >0    → max crates per vehicle.
    """
    df = df.copy().reset_index(drop=True)
    df.columns = df.columns.str.strip()

    # Normalize column names
    if "Tonnage" not in df.columns and "OrderKg" in df.columns:
        df = df.rename(columns={"OrderKg": "Tonnage"})

    crate_col = None
    if "TotalCrates" in df.columns:
        crate_col = "TotalCrates"
    elif "Crates" in df.columns:
        crate_col = "Crates"

    fc_lat = float(df["FC Latitude"].iloc[0])
    fc_lon = float(df["FC Longitude"].iloc[0])
    n = len(df)

    lats = [fc_lat] + df["Latitude"].astype(float).tolist()
    lons = [fc_lon] + df["Longitude"].astype(float).tolist()
    n_nodes = n + 1

    # ── Distance matrix ───────────────────────────────────────────────
    # Road distances from OSRM/ORS (or Haversine fallback).
    # Travel times are ALWAYS computed from distances ÷ configured speed_kmh
    # so the vehicle speed setting is respected regardless of routing source.
    routing_source = routing
    if routing == "osrm":
        try:
            dist, _ = fetch_osrm_matrix(lats, lons)   # only road distances used
        except Exception as exc:
            print(f"[optimizer] OSRM failed ({exc}), falling back to Haversine.")
            routing_source = "haversine (OSRM fallback)"
            routing = "haversine"

    if routing == "ors":
        try:
            dist, _ = fetch_ors_matrix(lats, lons, ors_api_key)   # only road distances used
        except Exception as exc:
            print(f"[optimizer] ORS failed ({exc}), falling back to Haversine.")
            routing_source = "haversine (ORS fallback)"
            routing = "haversine"

    if routing == "haversine":
        dist = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    dist[i][j] = haversine_km(lats[i], lons[i], lats[j], lons[j])
        if routing_source == routing:
            routing_source = "haversine"

    # Time matrix: always derived from road distances at the configured vehicle speed
    time_mat = np.round(dist / speed_kmh * 60).astype(int)

    time_windows = [(0, 1440)] + [parse_slot_minutes(row["Slot"]) for _, row in df.iterrows()]
    demands = [0] + [int(float(t)) for t in df["Tonnage"].tolist()]

    # ── Crate demands ────────────────────────────────────────────────
    if crate_col:
        crate_demands = [0] + [float(df[crate_col].iloc[i]) for i in range(n)]
    else:
        crate_demands = [0] * n_nodes

    # ── Service times per node (all 4 components happen at the customer) ──
    # service_time[node] = loading + unloading + buffer + waiting (capped at max_service_time_min)
    service_times = [0]  # depot has no service time
    for i in range(n):
        if crate_col:
            crates = float(df[crate_col].iloc[i])
            svc = (crates * (loading_sec_per_crate + unloading_sec_per_crate)) / 60.0 + buffer_time_min + waiting_time_min
        else:
            svc = buffer_time_min + waiting_time_min
        svc = min(svc, max_service_time_min)
        service_times.append(round(svc))

    return {
        "n_nodes":                n_nodes,
        "n_customers":            n,
        "num_vehicles":           num_vehicles,
        "depot":                  0,
        "lats":                   lats,
        "lons":                   lons,
        "dist_matrix":            dist,
        "time_matrix":            time_mat,
        "time_windows":           time_windows,
        "demands":                demands,
        "max_tonnage":            int(max_tonnage),
        "max_drops":              int(max_drops),
        "speed_kmh":              speed_kmh,
        "round_trip":             round_trip,
        "unique_slots":           unique_slots,
        "routing_source":         routing_source,
        "customers":              df,
        "fc_lat":                 fc_lat,
        "fc_lon":                 fc_lon,
        "crate_demands":          crate_demands,
        "max_crates":             int(max_crates),
        "loading_sec_per_crate":  loading_sec_per_crate,
        "unloading_sec_per_crate": unloading_sec_per_crate,
        "buffer_time_min":        buffer_time_min,
        "waiting_time_min":       waiting_time_min,
        "max_service_time_min":   max_service_time_min,
        "service_times":          service_times,
        "crate_col":              crate_col,
    }


# ─────────────────────── OR-Tools Solver ───────────────────────────

def solve_with_ortools(data: Dict) -> Optional[List[List[int]]]:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    manager = pywrapcp.RoutingIndexManager(
        data["n_nodes"], data["num_vehicles"], data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    dist_int = np.round(data["dist_matrix"] * 1000).astype(int)
    round_trip = data.get("round_trip", True)

    def dist_cb(fi, ti):
        from_node = manager.IndexToNode(fi)
        to_node   = manager.IndexToNode(ti)
        # One-way mode: returning to depot costs nothing
        if not round_trip and to_node == data["depot"]:
            return 0
        return int(dist_int[from_node][to_node])

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    # ── Time dimension with service time at each node ────────────────
    # time_cb(i→j) = service_time[i] + travel_time[i][j]
    # CumulVar at a node = arrival time; service happens after arrival.
    def time_cb(fi, ti):
        from_node = manager.IndexToNode(fi)
        to_node   = manager.IndexToNode(ti)
        svc = data["service_times"][from_node]
        return int(data["time_matrix"][from_node][to_node]) + svc

    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, 120, 1440, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    time_dim.SetGlobalSpanCostCoefficient(100)

    for node in range(1, data["n_nodes"]):
        idx = manager.NodeToIndex(node)
        start, end = data["time_windows"][node]
        time_dim.CumulVar(idx).SetRange(start, end)

    for v in range(data["num_vehicles"]):
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    # ── Tonnage capacity dimension ───────────────────────────────────
    def demand_cb(fi):
        return data["demands"][manager.IndexToNode(fi)]

    dem_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        dem_idx, 0, [data["max_tonnage"]] * data["num_vehicles"], True, "Capacity"
    )

    # ── Stops (max drop points) dimension ───────────────────────────
    def stop_cb(fi):
        return 0 if manager.IndexToNode(fi) == data["depot"] else 1

    stop_idx = routing.RegisterUnaryTransitCallback(stop_cb)
    routing.AddDimensionWithVehicleCapacity(
        stop_idx, 0, [data["max_drops"]] * data["num_vehicles"], True, "Stops"
    )

    # ── Crate capacity dimension (if enabled) ────────────────────────
    if data["max_crates"] > 0:
        def crate_cb(fi):
            return int(data["crate_demands"][manager.IndexToNode(fi)])

        crate_idx = routing.RegisterUnaryTransitCallback(crate_cb)
        routing.AddDimensionWithVehicleCapacity(
            crate_idx, 0, [data["max_crates"]] * data["num_vehicles"], True, "Crates"
        )

    # ── Unique-slot-per-vehicle constraint ───────────────────────────
    if data.get("unique_slots", False):
        from collections import defaultdict as _dd
        slot_buckets: Dict[str, List[int]] = _dd(list)
        for ci in range(data["n_customers"]):
            s = str(data["customers"].iloc[ci]["Slot"]).strip()
            slot_buckets[s].append(ci + 1)

        for dim_idx, (_, nodes) in enumerate(slot_buckets.items()):
            if len(nodes) < 2:
                continue
            ns = frozenset(nodes)
            def _make_cb(ns_=ns):
                def _cb(fi):
                    return 1 if manager.IndexToNode(fi) in ns_ else 0
                return _cb
            su_idx = routing.RegisterUnaryTransitCallback(_make_cb())
            routing.AddDimensionWithVehicleCapacity(
                su_idx, 0, [1] * data["num_vehicles"], True, f"US{dim_idx}"
            )

    penalty = 1_000_000_000
    for node in range(1, data["n_nodes"]):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 30

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None

    routes = []
    for v in range(data["num_vehicles"]):
        stops: List[int] = []
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != data["depot"]:
                stops.append(node - 1)
            idx = solution.Value(routing.NextVar(idx))
        if stops:
            routes.append(stops)
    return routes


# ────────────────────── Greedy Solver (fallback) ───────────────────

def solve_greedy(data: Dict) -> Optional[List[List[int]]]:
    unassigned = set(range(data["n_customers"]))
    routes: List[List[int]] = []
    unique_slots = data.get("unique_slots", False)
    customers = data["customers"]
    max_crates = data["max_crates"]

    for _ in range(data["num_vehicles"]):
        if not unassigned:
            break

        route: List[int] = []
        total_tonnage = 0
        total_crates = 0.0
        current_pos = 0
        current_time = 0
        route_slot_set: set = set()

        while len(route) < data["max_drops"] and unassigned:
            best_ci: Optional[int] = None
            best_score = float("inf")
            best_arrival = 0

            for ci in unassigned:
                node = ci + 1

                # Unique-slot constraint
                if unique_slots:
                    slot_str = str(customers.iloc[ci]["Slot"]).strip()
                    if slot_str in route_slot_set:
                        continue

                tonnage = data["demands"][node]
                if total_tonnage + tonnage > data["max_tonnage"]:
                    continue

                # Crate capacity constraint
                if max_crates > 0:
                    crates = data["crate_demands"][node]
                    if total_crates + crates > max_crates:
                        continue

                slot_start, slot_end = data["time_windows"][node]
                travel = data["time_matrix"][current_pos][node]
                arrival = current_time + travel

                if arrival > slot_end:
                    continue

                actual_arrival = max(arrival, slot_start)
                # Score = earliest we can DEPART this stop × 100 + road distance.
                # Using actual_arrival (not just slot_start) penalises early arrivals
                # that require long waits, producing tighter, more realistic schedules.
                departure = actual_arrival + data["service_times"][node]
                score = departure * 100 + data["dist_matrix"][current_pos][node]

                if score < best_score:
                    best_score = score
                    best_ci = ci
                    best_arrival = actual_arrival

            if best_ci is None:
                break

            route.append(best_ci)
            unassigned.remove(best_ci)
            node = best_ci + 1
            total_tonnage += data["demands"][node]
            if max_crates > 0:
                total_crates += data["crate_demands"][node]

            # Advance time: arrival + service time at this stop
            current_time = best_arrival + data["service_times"][node]
            current_pos = node
            if unique_slots:
                route_slot_set.add(str(customers.iloc[best_ci]["Slot"]).strip())

        if route:
            routes.append(route)

    return routes if routes else None


# ──────────────────────── Result Formatter ─────────────────────────

def format_results(
    routes: List[List[int]],
    data: Dict,
    vehicle_id_offset: int = 0,
) -> Dict:
    """
    Convert raw routes into output dataframes and map-ready structures.
    """
    customers   = data["customers"]
    speed_kmh   = data["speed_kmh"]
    round_trip  = data.get("round_trip", True)
    max_crates  = data["max_crates"]
    crate_col   = data.get("crate_col")
    all_served  = {ci for route in routes for ci in route}

    # Sort routes: vehicle with earliest first-stop slot departs first
    routes = sorted(
        routes,
        key=lambda stops: min(data["time_windows"][ci + 1][0] for ci in stops),
    )

    route_rows:    List[Dict] = []
    summary_rows:  List[Dict] = []
    route_structs: List[Dict] = []
    total_dist = 0.0

    for v_idx, stop_indices in enumerate(routes):
        if not stop_indices:
            continue

        actual_vid   = v_idx + vehicle_id_offset
        route_dist   = 0.0
        prev_node    = 0
        current_time = 0
        stops_data:  List[Dict] = []
        total_tonnage = sum(data["demands"][ci + 1] for ci in stop_indices)
        total_crates  = sum(data["crate_demands"][ci + 1] for ci in stop_indices)

        for stop_num, ci in enumerate(stop_indices, 1):
            node = ci + 1
            row  = customers.iloc[ci]
            lat  = float(row["Latitude"])
            lon  = float(row["Longitude"])

            leg_dist    = data["dist_matrix"][prev_node][node]
            travel_time = data["time_matrix"][prev_node][node]
            route_dist += leg_dist

            arrival_time = current_time + travel_time
            slot_start, slot_end = data["time_windows"][node]
            actual_arrival = max(arrival_time, slot_start)

            svc_time = data["service_times"][node]   # loading + unloading + buffer + waiting
            current_time = actual_arrival + svc_time
            prev_node    = node

            # Per-customer service time breakdown (all 4 components at the customer)
            crates_here   = data["crate_demands"][node]
            loading_min   = round((crates_here * data["loading_sec_per_crate"]) / 60.0, 1)
            unloading_min = round((crates_here * data["unloading_sec_per_crate"]) / 60.0, 1)
            buffer_min_   = data["buffer_time_min"]
            waiting_min_  = data["waiting_time_min"]
            overall_svc   = round(min(loading_min + unloading_min + buffer_min_ + waiting_min_, data["max_service_time_min"]), 1)

            stops_data.append({
                "stop_num":      stop_num,
                "customer_idx":  ci,
                "node":          node,
                "customer_id":   str(row.get("CustomerId", "")).strip(),
                "customer":      str(row.get("Customer", "")).strip(),
                "lat":           lat,
                "lon":           lon,
                "slot":          str(row.get("Slot", "")).strip(),
                "tonnage":       data["demands"][node],
                "crates":        crates_here,
                "loading_min":   loading_min,
                "unloading_min": unloading_min,
                "buffer_min":    buffer_min_,
                "waiting_min":   waiting_min_,
                "overall_svc":   overall_svc,
                "arrival_str":   minutes_to_hhmm(actual_arrival),
            })

        # ── Travel to next customer (blank for last stop) ─────────────
        for i, s in enumerate(stops_data):
            if i < len(stops_data) - 1:
                next_node = stops_data[i + 1]["node"]
                s["travel_to_next_min"] = int(data["time_matrix"][s["node"]][next_node])
                s["travel_to_next_km"]  = round(float(data["dist_matrix"][s["node"]][next_node]), 2)
            else:
                s["travel_to_next_min"] = None
                s["travel_to_next_km"]  = None

        # ── Return leg ──────────────────────────────────────────────
        if round_trip:
            route_dist += data["dist_matrix"][prev_node][0]

        total_dist += route_dist

        vehicle_label = f"Vehicle {actual_vid + 1}"
        trip_label    = f"Trip {actual_vid + 1}"
        route_type    = "Round Trip" if round_trip else "One-Way"

        for s in stops_data:
            base: Dict[str, Any] = {
                "Vehicle":    vehicle_label,
                "Trip":       trip_label,
                "RouteType":  route_type,
                "StopNumber": s["stop_num"],
                "TotalStops": len(stop_indices),
            }
            orig = customers.iloc[s["customer_idx"]]
            for col in customers.columns:
                base[col] = orig[col]
            base["EstimatedArrival"]        = s["arrival_str"]
            base["Crates"]                  = s["crates"]
            base["LoadingTime_min"]         = s["loading_min"]
            base["UnloadingTime_min"]       = s["unloading_min"]
            base["BufferTime_min"]          = s["buffer_min"]
            base["WaitingTime_min"]         = s["waiting_min"]
            base["OverallServiceTime_min"]  = s["overall_svc"]
            base["TravelToNext_min"]        = s["travel_to_next_min"]
            base["TravelToNext_km"]         = s["travel_to_next_km"]
            base["TotalTonnage_kg"]         = round(total_tonnage, 1)
            base["RouteTotalCrates"]        = round(total_crates, 1)
            base["RouteTotalDistance_km"]   = round(route_dist, 2)
            route_rows.append(base)

        summary_row: Dict[str, Any] = {
            "Vehicle":          vehicle_label,
            "Trip":             trip_label,
            "RouteType":        route_type,
            "StopCount":        len(stop_indices),
            "TotalTonnage_kg":  round(total_tonnage, 1),
            "TotalDistance_km": round(route_dist, 2),
            "EstDuration_hr":   round(route_dist / speed_kmh, 2),
        }
        total_service_min = sum(s["overall_svc"] for s in stops_data)
        summary_row["TotalCrates"]          = round(total_crates, 1)
        summary_row["TotalServiceTime_min"] = round(total_service_min, 1)
        summary_rows.append(summary_row)

        route_structs.append({
            "vehicle_id":     actual_vid,
            "stops":          stops_data,
            "total_distance": round(route_dist, 2),
            "round_trip":     round_trip,
        })

    return {
        "routes":         route_structs,
        "route_df":       pd.DataFrame(route_rows),
        "summary_df":     pd.DataFrame(summary_rows),
        "vehicles_used":  len(route_structs),
        "total_distance": round(total_dist, 2),
        "unserved":       data["n_customers"] - len(all_served),
        "routing_source": data.get("routing_source", "haversine"),
        "has_crates":     max_crates > 0 or bool(data.get("crate_col")),
    }


# ──────────────────── Inner Single-Group Solver ─────────────────────

def _solve_group(
    df: pd.DataFrame,
    num_vehicles: int,
    max_drops: int,
    max_tonnage: float,
    speed_kmh: float,
    use_ortools: bool,
    routing: str,
    ors_api_key: str,
    round_trip: bool,
    vehicle_id_offset: int = 0,
    unique_slots: bool = False,
    max_crates: int = 0,
    loading_sec_per_crate: float = 30.0,
    unloading_sec_per_crate: float = 45.0,
    buffer_time_min: float = 3.0,
    waiting_time_min: float = 5.0,
    max_service_time_min: float = 45.0,
) -> Optional[Dict]:
    """Solve VRP for a single customer group and return formatted results."""
    data = build_problem_data(
        df, num_vehicles, max_drops, max_tonnage,
        speed_kmh, routing, ors_api_key, round_trip, unique_slots,
        max_crates, loading_sec_per_crate, unloading_sec_per_crate,
        buffer_time_min, waiting_time_min, max_service_time_min,
    )
    routes = None

    if use_ortools:
        try:
            routes = solve_with_ortools(data)
        except ImportError:
            pass
        except Exception as exc:
            print(f"[optimizer] OR-Tools error: {exc}")

    if routes is None:
        routes = solve_greedy(data)

    if not routes:
        return None

    return format_results(routes, data, vehicle_id_offset=vehicle_id_offset)


# ──────────────── No-Overlapping-Slot Grouping Solver ──────────────

def _solve_no_overlap(
    df: pd.DataFrame,
    num_vehicles: int,
    max_drops: int,
    max_tonnage: float,
    speed_kmh: float,
    use_ortools: bool,
    routing: str,
    ors_api_key: str,
    round_trip: bool,
    max_crates: int = 0,
    loading_sec_per_crate: float = 30.0,
    unloading_sec_per_crate: float = 45.0,
    buffer_time_min: float = 3.0,
    waiting_time_min: float = 5.0,
    max_service_time_min: float = 45.0,
) -> Optional[Dict]:
    return _solve_group(
        df, num_vehicles, max_drops, max_tonnage,
        speed_kmh, use_ortools, routing, ors_api_key,
        round_trip, unique_slots=True,
        max_crates=max_crates,
        loading_sec_per_crate=loading_sec_per_crate,
        unloading_sec_per_crate=unloading_sec_per_crate,
        buffer_time_min=buffer_time_min,
        waiting_time_min=waiting_time_min,
        max_service_time_min=max_service_time_min,
    )


# ─────────────────────────── Entry Point ───────────────────────────

def solve_vrp(
    df: pd.DataFrame,
    num_vehicles: int,
    max_drops: int,
    max_tonnage: float,
    speed_kmh: float,
    use_ortools: bool         = True,
    routing: str              = "haversine",
    ors_api_key: str          = "",
    allow_overlapping_slots: bool = True,
    round_trip: bool          = True,
    max_crates: int           = 0,
    loading_sec_per_crate: float = 30.0,
    unloading_sec_per_crate: float = 45.0,
    buffer_time_min: float    = 3.0,
    waiting_time_min: float   = 5.0,
    max_service_time_min: float = 45.0,
) -> Optional[Dict]:
    """
    Main entry point.
    max_crates: 0 disables crate constraint. >0 enforces per-vehicle crate limit.
    """
    kwargs = dict(
        max_crates=max_crates,
        loading_sec_per_crate=loading_sec_per_crate,
        unloading_sec_per_crate=unloading_sec_per_crate,
        buffer_time_min=buffer_time_min,
        waiting_time_min=waiting_time_min,
        max_service_time_min=max_service_time_min,
    )

    if not allow_overlapping_slots:
        return _solve_no_overlap(
            df, num_vehicles, max_drops, max_tonnage,
            speed_kmh, use_ortools, routing, ors_api_key, round_trip,
            **kwargs,
        )

    return _solve_group(
        df, num_vehicles, max_drops, max_tonnage,
        speed_kmh, use_ortools, routing, ors_api_key, round_trip,
        **kwargs,
    )

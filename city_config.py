"""
city_config.py - starter MDCs + vehicles for a city that has no constraint
workbook, built from where its customers are. Used by the app (cities found
in the database without a default_config_<city>.xlsx) and by
build_city_config.py (to save one as a file).

  * MDCs     - split customers into N geographic groups (k-means) and put each
               MDC at the group's most central customer location (the medoid),
               so it always sits on land / on the road network. Rounded to 3 dp.
  * Vehicles - Bengaluru's vehicle types and costs:
                 Bike : 4 drops, 20 kg, Rs 0 fixed + Rs 19/km
                 EV   : 10 drops, 700 kg, Rs 1600 fixed + Rs 0/km
               bikes per MDC = max(6, ceil(1.5 x customers / 4)); 1 EV per MDC
               (more if the group has many >20 kg customers).
               One-way caps from real road km (OSRM) to the group's farthest
               customer: bikes 25-50 km, EV = bike cap + 10 (max 60).
"""
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

OSRM_TABLE = "http://router.project-osrm.org/table/v1/driving/"


def hav_matrix(lat, lon):
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    a = np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def kmeans(xy: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(20):                                  # restarts, keep the tightest
        cen = xy[rng.choice(len(xy), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((xy[:, None] - cen[None]) ** 2).sum(-1), axis=1)
            new = np.array([xy[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
            if np.allclose(new, cen):
                break
            cen = new
        sse = ((xy - cen[lab]) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, lab)
    return best[1]


def road_km_from(origin, pts) -> np.ndarray:
    """Driving km from origin to each point (OSRM); straight line x1.4 if OSRM fails."""
    pts = list(pts)
    try:
        coords = ";".join(f"{lo:.6f},{la:.6f}" for la, lo in [origin] + pts)
        r = requests.get(OSRM_TABLE + coords, params={"sources": "0", "annotations": "distance"},
                         timeout=60)
        r.raise_for_status()
        return np.array([np.nan if v is None else v / 1000 for v in r.json()["distances"][0][1:]])
    except Exception:
        lat = np.array([origin[0]] + [p[0] for p in pts])
        lon = np.array([origin[1]] + [p[1] for p in pts])
        return hav_matrix(lat, lon)[0, 1:] * 1.4


def direction(a, b) -> str:
    """Which side of the city MDC a is on, relative to MDC b."""
    dlat, dlon = a[0] - b[0], (a[1] - b[1]) * math.cos(math.radians(a[0]))
    return ("North" if dlat > 0 else "South") if abs(dlat) >= abs(dlon) else ("East" if dlon > 0 else "West")


def ceil5(x) -> int:
    return int(math.ceil(x / 5.0) * 5)


def build_starter(cust: pd.DataFrame, n_mdcs: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """cust: cleaned customers (CustomerId, Latitude, Longitude, Tonnage).
    Returns (mdcs, vehicles, log lines) in the app's MDC / vehicle table layout."""
    n_mdcs = max(1, min(n_mdcs, len(cust)))
    lat, lon = cust.Latitude.to_numpy(float), cust.Longitude.to_numpy(float)
    xy = np.c_[lat * 111.0, lon * 111.0 * math.cos(math.radians(lat.mean()))]
    lab = kmeans(xy, n_mdcs) if n_mdcs > 1 else np.zeros(len(cust), int)
    dist = hav_matrix(lat, lon)

    groups = []
    for j in range(n_mdcs):
        idx = np.where(lab == j)[0]
        if not len(idx):
            continue
        medoid = idx[np.argmin(dist[np.ix_(idx, idx)].sum(axis=1))]
        groups.append((idx, (round(float(lat[medoid]), 3), round(float(lon[medoid]), 3))))
    groups.sort(key=lambda g: -len(g[0]))                 # MDC1 = the bigger group

    mdcs, vehicles, log = [], [], []
    for j, (idx, site) in enumerate(groups, start=1):
        code = f"MDC{j}"
        others = [g[1] for k, g in enumerate(groups, start=1) if k != j]
        where = direction(site, others[0]) if others else "Central"
        mdcs.append({"MDC": code, "Site": f"{where} (auto-placed)",
                     "Latitude": site[0], "Longitude": site[1]})

        sub = cust.iloc[idx]
        road = road_km_from(site, zip(sub.Latitude, sub.Longitude))
        far = float(np.nanmax(road)) if np.isfinite(road).any() else 25.0
        bike_cap = min(50, max(25, ceil5(far)))
        ev_cap = min(60, bike_cap + 10)
        n_bikes = max(6, math.ceil(1.5 * len(sub) / 4))
        n_ev = max(1, math.ceil(int((sub.Tonnage > 20).sum()) / 10))
        vehicles += [{"Active": True, "MDC": code, "Vehicle": f"EV{i}", "DropPoints": 10,
                      "MaxDistance_km": float(ev_cap), "MaxTonnage_kg": 700.0, "FixedCost_Rs": 1600.0,
                      "VariableCost_Rs_per_km": 0.0, "ShiftStart": "", "ShiftEnd": ""}
                     for i in range(1, n_ev + 1)]
        vehicles += [{"Active": True, "MDC": code, "Vehicle": f"Bike{i}", "DropPoints": 4,
                      "MaxDistance_km": float(bike_cap), "MaxTonnage_kg": 20.0, "FixedCost_Rs": 0.0,
                      "VariableCost_Rs_per_km": 19.0, "ShiftStart": "", "ShiftEnd": ""}
                     for i in range(1, n_bikes + 1)]
        log.append(f"{code} {where} {site}: {len(sub)} customers, farthest {far:.1f} km by road "
                   f"-> {n_ev} EV ({ev_cap} km) + {n_bikes} bikes ({bike_cap} km)")
        if far > ev_cap:
            log.append(f"   {int((road > ev_cap).sum())} customer(s) beyond every vehicle's cap")
    return pd.DataFrame(mdcs), pd.DataFrame(vehicles), log

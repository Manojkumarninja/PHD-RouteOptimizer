"""
mdc_streamlit_app.py - daily MDC route planner (Streamlit) over the
multi-depot engine in simulate_bangalore.py.

  * Customers + tonnage come only from the PHD_MDCRO_Base table, and only
    TODAY's rows. "Refresh data" re-queries, so new orders show up
    immediately. (If the DB isn't connected yet, the sample day from
    "Banglore Constraint1.xlsx" is shown as a clearly-labelled preview.)
  * MDCs and vehicles default to Case 2 of "Banglore Constraint1.xlsx"
    (MDC1 + MDC2 Boomanalli, one row per vehicle exactly as in the sheet).
    Edit them in the app, or download them as Excel, change them there and
    upload the file back.
  * Every customer's delivery window is editable too.
  * Routes are one-way (MDC -> last customer), solved with OR-Tools on
    OpenStreetMap road distances (OSRM, free, no key), and drawn on an
    OpenStreetMap map along the actual roads.

Database: put the connection in .streamlit/secrets.toml (see
secrets.toml.example next to this file).

Run:  streamlit run mdc_streamlit_app.py
"""
import io
import os
import re
import sys
import math
import hashlib
from datetime import datetime, date, time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate_bangalore as sim  # noqa: E402  (read_constraints / solve_case)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(APP_DIR, "default_config.xlsx")   # MDCs + vehicles, committed with the app
DEFAULTS_BOOK = os.environ.get("PHD_SAMPLE_BOOK",                 # local only: Case 2 + sample day
                               r"D:\PHD Clustering\Banglore Constraint1.xlsx")
DEFAULTS_CASE = "Case2"
DB_CONN_NAME = "phd_db"
DEFAULT_QUERY = "SELECT * FROM PHD_MDCRO_Base WHERE DATE(DeliveryDate) = :today"
DEFAULT_WINDOW = "9:30-13:30"
DEFAULT_CITY_ID = 48757          # Bengaluru in PHD data
OUT_OF_CITY_KM = 60              # farther than this from the day's median = bad location
VEHICLE_PENALTY_RS = 10_000      # objective-only, see sim.solve_case docstring
OSRM_ROUTE = "http://router.project-osrm.org/route/v1/driving/"

MDC_COLS = ["MDC", "Site", "Latitude", "Longitude"]
FLEET_COLS = ["Active", "MDC", "Vehicle", "DropPoints", "MaxDistance_km", "MaxTonnage_kg",
              "FixedCost_Rs", "VariableCost_Rs_per_km", "ShiftStart", "ShiftEnd"]
FLEET_REQUIRED = ["MDC", "Vehicle", "DropPoints", "MaxDistance_km", "MaxTonnage_kg"]

VEHICLE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#9A6324", "#469990", "#800000",
    "#808000", "#000075", "#a9a9a9", "#ff7f50", "#40e0d0",
    "#ffe119", "#4169e1", "#da70d6", "#2e8b57", "#b8860b",
]
MDC_ICON_COLORS = ["red", "blue", "green", "purple", "orange",
                   "darkred", "cadetblue", "black", "pink", "gray"]

# DB column names vary - map anything sensible onto the solver's names
COLUMN_ALIASES = {
    "customerid": "CustomerId", "custid": "CustomerId",
    "customer": "Customer", "customername": "Customer", "name": "Customer",
    "latitude": "Latitude", "lat": "Latitude",
    "longitude": "Longitude", "lon": "Longitude", "lng": "Longitude", "long": "Longitude",
    "tonnage": "Tonnage", "orderkg": "Tonnage", "kg": "Tonnage", "weight": "Tonnage",
    "quantitykg": "Tonnage",
    "deliverywindow": "DeliveryWindow", "slot": "DeliveryWindow", "timeslot": "DeliveryWindow",
    "deliveryslot": "DeliveryWindow",
    "deliverydate": "DeliveryDate", "date": "DeliveryDate", "orderdate": "DeliveryDate",
    "cityid": "CityId",
}
REQUIRED = ["CustomerId", "Customer", "Latitude", "Longitude", "Tonnage"]


# ───────────────────────────── time helpers ─────────────────────────────

def parse_hhmm(txt) -> Optional[int]:
    """'09:30' / '9:30' -> 570 minutes; blank -> None; invalid -> raises."""
    if txt is None or (isinstance(txt, float) and math.isnan(txt)) or str(txt).strip() == "":
        return None
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*", str(txt))
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError(f"'{txt}' is not HH:MM")
    return int(m.group(1)) * 60 + int(m.group(2))


def as_hhmm_text(v) -> str:
    """Excel turns a typed 09:30 into a time - bring it back to 'HH:MM' text."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or (not isinstance(v, str) and pd.isna(v)):
        return ""
    if isinstance(v, (dtime, datetime, pd.Timestamp)):
        return f"{v.hour:02d}:{v.minute:02d}"
    s = str(v).strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", s)
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else s


def window_ok(w) -> bool:
    m = re.findall(r"(\d{1,2}):(\d{2})", str(w))
    if len(m) != 2:
        return False
    (h1, m1), (h2, m2) = m
    a, b = int(h1) * 60 + int(m1), int(h2) * 60 + int(m2)
    return int(h1) < 24 and int(h2) < 24 and int(m1) < 60 and int(m2) < 60 and a < b


def as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "no", "n", "0", "")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return True             # blank = active
    return bool(v)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# ───────────────────────────── customer data ─────────────────────────────

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        key = re.sub(r"[^a-z]", "", str(c).lower())
        if key in COLUMN_ALIASES and COLUMN_ALIASES[key] not in ren.values():
            ren[c] = COLUMN_ALIASES[key]
    return df.rename(columns=ren)


def keep_today(raw: pd.DataFrame, today: date) -> Tuple[pd.DataFrame, Optional[Tuple[str, pd.DataFrame]]]:
    """Belt and braces: whatever the query returns, keep only today's rows."""
    df = normalise_columns(raw)
    if "DeliveryDate" not in df.columns:
        return df, None
    d = pd.to_datetime(df.DeliveryDate, errors="coerce").dt.date
    other = d != today
    note = (f"Rows not dated today ({today:%d %b %Y}) - excluded", df[other]) if other.any() else None
    return df[~other], note


def prepare_customers(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, pd.DataFrame]], List[str]]:
    """Clean + aggregate to one row per customer. Every exclusion is reported."""
    df = normalise_columns(raw)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return pd.DataFrame(), [], missing

    notes: List[Tuple[str, pd.DataFrame]] = []
    df = df.copy()
    for c in ("Latitude", "Longitude", "Tonnage"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "DeliveryWindow" not in df.columns:
        df["DeliveryWindow"] = DEFAULT_WINDOW
    df["DeliveryWindow"] = df["DeliveryWindow"].fillna(DEFAULT_WINDOW).astype(str).str.strip()

    bad = df[["Latitude", "Longitude", "Tonnage"]].isna().any(axis=1)
    if bad.any():
        notes.append(("Missing / non-numeric Latitude, Longitude or Tonnage - excluded", df[bad]))
    df = df[~bad]

    # swapped lat/lon (lat should be ~8-30, lon ~68-90 in India)
    swap = (df.Latitude > 45) & (df.Longitude < 45)
    if swap.any():
        notes.append(("Latitude/Longitude were swapped - fixed automatically", df[swap].copy()))
        df.loc[swap, ["Latitude", "Longitude"]] = df.loc[swap, ["Longitude", "Latitude"]].values

    bad = ~df.Latitude.between(6, 38) | ~df.Longitude.between(68, 98)
    if bad.any():
        notes.append(("Coordinates outside India - excluded", df[bad]))
    df = df[~bad]

    # placeholder / mistyped locations (e.g. 12.0, 77.0) far outside the city
    if len(df) >= 5:
        med_la, med_lo = df.Latitude.median(), df.Longitude.median()
        km = np.hypot((df.Latitude - med_la) * 111.0,
                      (df.Longitude - med_lo) * 111.0 * math.cos(math.radians(med_la)))
        far = km > OUT_OF_CITY_KM
        if far.any():
            notes.append((f"Location more than {OUT_OF_CITY_KM} km from the city - excluded "
                          f"(check the customer's lat/long in the source data)",
                          df[far].assign(km_from_city=km[far].round(1))))
        df = df[~far]

    # several order lines for the same customer -> one stop, tonnage summed
    n_lines = len(df)
    agg = (df.groupby("CustomerId", sort=False)
             .agg(Customer=("Customer", "first"), Latitude=("Latitude", "first"),
                  Longitude=("Longitude", "first"), Tonnage=("Tonnage", "sum"),
                  DeliveryWindow=("DeliveryWindow", "first"))
             .reset_index())
    if len(agg) < n_lines:
        dup = df[df.duplicated("CustomerId", keep=False)].sort_values("CustomerId")
        notes.append((f"{n_lines} order lines merged into {len(agg)} customers (tonnage summed)", dup))
    agg["Customer"] = agg["Customer"].astype(str)
    return agg, notes, []


def db_configured() -> bool:
    try:
        conn = st.secrets.get("connections", {}).get(DB_CONN_NAME, {})
    except Exception:          # no secrets.toml at all
        return False
    if conn.get("host"):       # separate host / username / password fields
        return not str(conn.get("host")).startswith("YOUR_")
    url = str(conn.get("url", ""))
    # the shipped secrets.toml still has its placeholders -> not connected yet
    return bool(url) and not any(p in url for p in ("USERNAME", "PASSWORD@HOST", "/DATABASE"))


def load_from_db(today_iso: str) -> pd.DataFrame:
    conn = st.connection(DB_CONN_NAME, type="sql")
    try:
        query = st.secrets.get("phd_mdcro", {}).get("query", DEFAULT_QUERY)
    except Exception:
        query = DEFAULT_QUERY
    return conn.query(query, params={"today": today_iso}, ttl=600)


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_excel(DEFAULTS_BOOK, sheet_name="CustomerBase")


# ───────────────────────────── MDC / vehicle config ─────────────────────────────

def _builtin_case2():
    """Same as Banglore Constraint1.xlsx Case 2, used only if the file is missing."""
    mdcs = [("MDC1", "MDC1", 13.0452978, 77.741736), ("MDC2", "Boomanalli", 12.90419, 77.626542)]
    veh = [("MDC1", f"EV{i}", 10, 40.0, 700.0, 1600.0, 0.0) for i in (1, 2)]
    veh += [("MDC1", f"Bike{i}", 4 if i % 2 else 3, 35.0, 20.0, 0.0, 19.0) for i in range(1, 20)]
    veh += [("MDC2", f"Bike{i}", 4, 25.0, 20.0, 0.0, 19.0) for i in range(1, 21) if i != 6]
    return mdcs, [dict(zip(["MDC", "Vehicle", "DropPoints", "MaxDistance_km", "MaxTonnage_kg",
                            "FixedCost_Rs", "VariableCost_Rs_per_km"], v)) for v in veh]


@st.cache_data(show_spinner=False)
def default_tables() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if os.path.exists(DEFAULT_CONFIG):
        with open(DEFAULT_CONFIG, "rb") as fh:
            mdc_df, fleet, err = read_config_upload(fh.read())
        if not err:
            return mdc_df, fleet, os.path.basename(DEFAULT_CONFIG)
    try:
        cfg = sim.read_constraints(DEFAULTS_BOOK)[DEFAULTS_CASE]
        mdcs = [(c, cfg["sites"].get(c, c), la, lo) for c, la, lo in cfg["depots"]]
        vehicles = cfg["vehicles"]
        src = f"{os.path.basename(DEFAULTS_BOOK)} - {DEFAULTS_CASE}"
    except Exception:
        mdcs, vehicles = _builtin_case2()
        src = "built-in copy of Banglore Constraint1.xlsx Case 2"
    mdc_df = pd.DataFrame(mdcs, columns=MDC_COLS)
    fleet = pd.DataFrame(vehicles)
    fleet.insert(0, "Active", True)
    fleet["ShiftStart"], fleet["ShiftEnd"] = "", ""
    return mdc_df, fleet[FLEET_COLS], src


def config_workbook(mdcs: pd.DataFrame, fleet: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        mdcs[MDC_COLS].to_excel(xl, sheet_name="MDCs", index=False)
        f = fleet.reindex(columns=FLEET_COLS).copy()
        f["ShiftStart"] = f["ShiftStart"].map(as_hhmm_text)
        f["ShiftEnd"] = f["ShiftEnd"].map(as_hhmm_text)
        f.to_excel(xl, sheet_name="Vehicles", index=False)
        pd.DataFrame({"How to use this file": [
            "Edit the MDCs and Vehicles sheets, save, and upload this file back into the app.",
            "MDCs: one row per MDC - code (MDC1, MDC2, ...), site name, latitude, longitude.",
            "Vehicles: one row per vehicle. Add a row to add a vehicle, delete a row to remove it,",
            "  or set Active to FALSE to keep it on file but leave it out of today's plan.",
            "MDC must match a code on the MDCs sheet. Vehicle names must be unique per MDC.",
            "DropPoints = max customers per trip (whole number).",
            "MaxDistance_km = ONE-WAY cap, MDC -> last customer (no return leg).",
            "MaxTonnage_kg = load capacity. FixedCost_Rs = per vehicle used per day.",
            "VariableCost_Rs_per_km = per one-way km.",
            "ShiftStart / ShiftEnd = optional HH:MM, e.g. 09:30. Blank = follow customer windows.",
            "Keep the column headers exactly as they are.",
        ]}).to_excel(xl, sheet_name="ReadMe", index=False)
    return buf.getvalue()


def _match_columns(df: pd.DataFrame, wanted: List[str]) -> pd.DataFrame:
    key = lambda c: re.sub(r"[^a-z]", "", str(c).lower())
    lookup = {key(w): w for w in wanted}
    return df.rename(columns={c: lookup[key(c)] for c in df.columns if key(c) in lookup})


def read_config_upload(data: bytes) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]:
    try:
        xl = pd.ExcelFile(io.BytesIO(data))
    except Exception as exc:
        return None, None, [f"Not a readable Excel file ({exc})"]
    missing = [s for s in ("MDCs", "Vehicles") if s not in xl.sheet_names]
    if missing:
        return None, None, [f"Missing sheet(s): {', '.join(missing)} - start from the downloaded "
                            f"config file so the layout matches"]
    mdcs = _match_columns(pd.read_excel(xl, "MDCs"), MDC_COLS)
    fleet = _match_columns(pd.read_excel(xl, "Vehicles"), FLEET_COLS)
    errors = []
    for name, df, req in (("MDCs", mdcs, MDC_COLS[:1] + MDC_COLS[2:]), ("Vehicles", fleet, FLEET_REQUIRED)):
        miss = [c for c in req if c not in df.columns]
        if miss:
            errors.append(f"Sheet '{name}' is missing column(s): {', '.join(miss)}")
    if errors:
        return None, None, errors
    if "Site" not in mdcs.columns:
        mdcs["Site"] = mdcs["MDC"]
    mdcs = mdcs[MDC_COLS].dropna(how="all").reset_index(drop=True)
    fleet = fleet.reindex(columns=FLEET_COLS).dropna(how="all", subset=FLEET_REQUIRED).reset_index(drop=True)
    fleet["Active"] = fleet["Active"].map(as_bool)
    for c in ("ShiftStart", "ShiftEnd"):
        fleet[c] = fleet[c].map(as_hhmm_text)
    for c in ("FixedCost_Rs", "VariableCost_Rs_per_km"):
        fleet[c] = pd.to_numeric(fleet[c], errors="coerce").fillna(0.0)
    fleet["MDC"] = fleet["MDC"].astype(str).str.strip().str.upper()
    fleet["Vehicle"] = fleet["Vehicle"].astype(str).str.strip()
    return mdcs, fleet, []


def build_config(mdcs: pd.DataFrame, fleet: pd.DataFrame) -> Tuple[Optional[Dict], List[str]]:
    errors = []
    depots, sites = [], {}
    for i, r in mdcs.reset_index(drop=True).iterrows():
        code = "" if pd.isna(r.get("MDC")) else str(r.get("MDC")).strip().upper()
        if not code:
            continue
        try:
            la, lo = float(r.Latitude), float(r.Longitude)
            if not (6 <= la <= 38 and 68 <= lo <= 98):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"MDC row {i + 1} ({code}): latitude/longitude missing or outside India")
            continue
        if code in sites:
            errors.append(f"MDC '{code}' appears twice")
            continue
        depots.append((code, la, lo))
        site = r.get("Site")
        sites[code] = code if site is None or pd.isna(site) or not str(site).strip() else str(site).strip()
    if not depots:
        errors.append("Add at least one MDC")

    rows, seen = [], set()
    for i, r in fleet.reset_index(drop=True).iterrows():
        home = "" if pd.isna(r.get("MDC")) else str(r.get("MDC")).strip().upper()
        name = "" if pd.isna(r.get("Vehicle")) else str(r.get("Vehicle")).strip()
        if not home and not name:
            continue
        if not as_bool(r.get("Active", True)):
            continue
        where = f"Vehicle row {i + 1} ({name or '?'} @ {home or '?'})"
        if not name:
            errors.append(f"{where}: give the vehicle a name")
            continue
        try:
            drops_f = float(r.DropPoints)
            dkm, ton = float(r.MaxDistance_km), float(r.MaxTonnage_kg)
            fixed = 0.0 if pd.isna(r.FixedCost_Rs) else float(r.FixedCost_Rs)
            var = 0.0 if pd.isna(r.VariableCost_Rs_per_km) else float(r.VariableCost_Rs_per_km)
            if any(map(math.isnan, (drops_f, dkm, ton))):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{where}: Max drops, Max one-way km and Max kg must be filled in")
            continue
        if drops_f != int(drops_f):
            errors.append(f"{where}: Max drops must be a whole number (got {drops_f:g})")
            continue
        drops = int(drops_f)
        if home not in sites:
            errors.append(f"{where}: MDC '{home}' is not in the MDC table")
            continue
        if (home, name.lower()) in seen:
            errors.append(f"{where}: vehicle name used twice at {home}")
            continue
        seen.add((home, name.lower()))
        if drops < 1 or dkm <= 0 or ton <= 0 or fixed < 0 or var < 0:
            errors.append(f"{where}: Max drops ≥ 1, distance and kg > 0, costs ≥ 0")
            continue
        try:
            s, e = parse_hhmm(r.get("ShiftStart")), parse_hhmm(r.get("ShiftEnd"))
        except ValueError as exc:
            errors.append(f"{where}: shift time {exc}")
            continue
        if s is not None and e is not None and s >= e:
            errors.append(f"{where}: shift start must be before shift end")
            continue
        label = re.sub(r"\s*\d+$", "", name).strip() or "Veh"
        rows.append((label, 1, drops, dkm, ton, home, fixed, var, s, e, name))
    if not rows:
        errors.append("Add at least one active vehicle")
    if errors:
        return None, errors
    return {"depots": depots, "sites": sites, "fleet": rows}, []


def unserved_reason(row, cfg: Dict) -> str:
    heaviest = max(f[4] for f in cfg["fleet"])
    if row.Tonnage > heaviest:
        return f"{row.Tonnage:g} kg is more than any vehicle can carry (max {heaviest:g} kg)."
    best = None
    for f in cfg["fleet"]:
        home, dkm, ton = f[5], f[3], f[4]
        if row.Tonnage > ton:
            continue
        la, lo = next((d[1], d[2]) for d in cfg["depots"] if d[0] == home)
        d = haversine_km(row.Latitude, row.Longitude, la, lo)
        if best is None or dkm - d > best[0]:
            best = (dkm - d, home, d, dkm, f[0])
    margin, home, d, cap, label = best
    if margin < 0:
        return (f"Too far: nearest option is a {label} from {home}, {d:.1f} km away in a straight "
                f"line, but its one-way cap is {cap:g} km. Raise that vehicle's Max one-way km.")
    return (f"Reachable ({label} from {home}: {d:.1f} km, cap {cap:g} km) - likely the delivery "
            f"window, drop limits or too few vehicles. Add vehicles or give the solver more time.")


# ───────────────────────────── map ─────────────────────────────

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def road_geometry(points: Tuple[Tuple[float, float], ...]) -> Optional[List[List[float]]]:
    """Road-following path through the points, from OSRM (OpenStreetMap, free)."""
    try:
        coords = ";".join(f"{lo:.6f},{la:.6f}" for la, lo in points)
        r = requests.get(OSRM_ROUTE + coords, params={"overview": "full", "geometries": "geojson"},
                         timeout=15)
        js = r.json()
        if r.ok and js.get("code") == "Ok":
            return [[la, lo] for lo, la in js["routes"][0]["geometry"]["coordinates"]]
    except Exception:
        pass
    return None


def build_map(res: Dict, cfg: Dict, cust: pd.DataFrame) -> Tuple[folium.Map, int]:
    depots = cfg["depots"]
    lat_all = [d[1] for d in depots] + cust.Latitude.tolist()
    lon_all = [d[2] for d in depots] + cust.Longitude.tolist()
    m = folium.Map(location=[float(np.mean(lat_all)), float(np.mean(lon_all))],
                   zoom_start=11, tiles="OpenStreetMap")
    m.fit_bounds([[min(lat_all), min(lon_all)], [max(lat_all), max(lon_all)]])

    mdc_color = {d[0]: MDC_ICON_COLORS[i % len(MDC_ICON_COLORS)] for i, d in enumerate(depots)}
    for code, la, lo in depots:
        site = cfg["sites"].get(code, code)
        label = code if site == code else f"{code} ({site})"
        folium.Marker([la, lo], tooltip=label,
                      popup=folium.Popup(f"<b>{label}</b>", max_width=180),
                      icon=folium.Icon(color=mdc_color[code], icon="home", prefix="fa")).add_to(m)

    fallbacks = 0
    legend = []
    stops = res["stops"]
    if len(stops):
        for i, (veh, g) in enumerate(stops.groupby("Vehicle", sort=False)):
            g = g.sort_values("StopNumber")
            color = VEHICLE_COLORS[i % len(VEHICLE_COLORS)]
            home = next((la, lo) for c, la, lo in depots if c == g.MDC.iloc[0])
            pts = (home,) + tuple((float(a), float(b)) for a, b in zip(g.Latitude, g.Longitude))
            path = road_geometry(pts)
            if path is None:
                fallbacks += 1
            km = g.LegDistance_km.sum()
            folium.PolyLine(path or [list(p) for p in pts], color=color, weight=4, opacity=0.85,
                            dash_array=None if path else "6 6",
                            tooltip=f"{veh} | {len(g)} stops | {km:.1f} km one-way").add_to(m)
            for _, s in g.iterrows():
                name = str(s.Customer)
                popup = (f"<b>{veh}</b> · stop {s.StopNumber} of {s.TotalStops}<br>"
                         f"{name} (ID {s.CustomerId})<br>MDC: {s.MDC}<br>"
                         f"{s.Tonnage:g} kg · window {s.DeliveryWindow}<br>"
                         f"ETA <b>{s.EstimatedArrival}</b> · leg {s.LegDistance_km} km")
                folium.Marker(
                    [s.Latitude, s.Longitude],
                    tooltip=f"{veh} · stop {s.StopNumber} · {name[:30]}",
                    popup=folium.Popup(popup, max_width=300),
                    icon=folium.DivIcon(html=(
                        f'<div style="font-size:11px;font-weight:bold;color:white;background:{color};'
                        f'border:2px solid white;border-radius:50%;width:22px;height:22px;'
                        f'text-align:center;line-height:18px;box-shadow:0 0 3px #555;">'
                        f'{s.StopNumber}</div>'), icon_size=(22, 22), icon_anchor=(11, 11)),
                ).add_to(m)
            legend.append((color, veh, len(g), km))

    for idx in res["unserved"]:
        r = cust.iloc[idx]
        folium.Marker([r.Latitude, r.Longitude], tooltip=f"UNSERVED: {r.Customer}",
                      icon=folium.Icon(color="red", icon="times", prefix="fa")).add_to(m)

    items = "".join(
        f'<div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
        f'background:{c};margin-right:6px;"></span><b>{v}</b> {n} stops · {km:.1f} km</div>'
        for c, v, n, km in legend)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
        'padding:10px 14px;border-radius:8px;border:1px solid #bbb;font-size:12px;'
        f'max-height:300px;overflow-y:auto;"><b>Vehicles</b>{items}</div>'))
    return m, fallbacks


# ───────────────────────────── results ─────────────────────────────

def assignment_table(res: Dict, cust: pd.DataFrame) -> pd.DataFrame:
    stops = res["stops"]
    cols = ["CustomerId", "Customer", "MDC", "Vehicle", "VehicleType", "StopNumber",
            "EstimatedArrival", "DeliveryWindow", "Tonnage", "LegDistance_km", "CumulativeDistance_km"]
    served = (stops[cols] if len(stops) else pd.DataFrame(columns=cols)).rename(columns={"MDC": "Assigned_MDC"})
    unserved = cust.iloc[res["unserved"]][["CustomerId", "Customer", "DeliveryWindow", "Tonnage"]].copy()
    unserved["Assigned_MDC"] = "UNSERVED"
    return pd.concat([served, unserved], ignore_index=True)


def mdc_summary(res: Dict, cfg: Dict) -> pd.DataFrame:
    s = res["summary"]
    rows = []
    for code, la, lo in cfg["depots"]:
        g = s[s.MDC == code] if len(s) else s
        rows.append({"MDC": code, "Site": cfg["sites"].get(code, code),
                     "Vehicles": len(g), "Customers": int(g.Stops.sum()) if len(g) else 0,
                     "Tonnage_kg": round(g.Tonnage.sum(), 2) if len(g) else 0,
                     "OneWay_km": round(g.OneWayDistance_km.sum(), 1) if len(g) else 0,
                     "Cost_Rs": round(g.TotalCost.sum(), 0) if len(g) else 0})
    return pd.DataFrame(rows)


def results_workbook(res, cfg, cust, mdcs, fleet) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        assignment_table(res, cust).to_excel(xl, sheet_name="Customer_Assignment", index=False)
        mdc_summary(res, cfg).to_excel(xl, sheet_name="MDC_Summary", index=False)
        res["summary"].to_excel(xl, sheet_name="Vehicle_Summary", index=False)
        res["stops"].to_excel(xl, sheet_name="Routes", index=False)
        cust.to_excel(xl, sheet_name="Input_Customers", index=False)
        mdcs.to_excel(xl, sheet_name="Input_MDCs", index=False)
        fleet.to_excel(xl, sheet_name="Input_Vehicles", index=False)
    return buf.getvalue()


def signature(*objs) -> str:
    h = hashlib.md5()
    for o in objs:
        h.update(pd.util.hash_pandas_object(o.astype(str), index=False).values.tobytes()
                 if isinstance(o, pd.DataFrame) else repr(o).encode())
    return h.hexdigest()


# ───────────────────────────── app ─────────────────────────────

def main():
    st.set_page_config(page_title="MDC Route Planner", layout="wide", page_icon="🚚")
    ss = st.session_state
    ss.setdefault("win_over", {})
    ss.setdefault("incl_over", {})
    ss.setdefault("cust_ver", 0)
    ss.setdefault("cfg_ver", 0)
    today = date.today()

    # ── sidebar: today's customers ──
    st.sidebar.header("Today's customers")
    have_db = db_configured()
    if have_db:
        st.sidebar.caption(f"From **PHD_MDCRO_Base**, {today:%d %b %Y} only.")
    elif os.path.exists(DEFAULTS_BOOK):
        st.sidebar.warning("Database not connected - showing the sample day as a preview. "
                           "Add the connection in `.streamlit/secrets.toml` "
                           "(see `secrets.toml.example`).")
    else:
        st.sidebar.error("Database not connected. Add the `[connections.phd_db]` section to the "
                         "app's secrets (Streamlit Cloud: app → Settings → Secrets).")
    if st.sidebar.button("🔄 Refresh data", width="stretch"):
        st.cache_data.clear()
        ss.pop("fetched_at", None)
        st.rerun()
    if ss.get("fetched_at", ("",))[0] != today.isoformat():
        ss.fetched_at = (today.isoformat(), datetime.now().strftime("%H:%M:%S"))

    raw, source_label, date_note = None, "", None
    try:
        if have_db:
            raw, date_note = keep_today(load_from_db(today.isoformat()), today)
            source_label = f"PHD_MDCRO_Base · {today:%d %b %Y}"
        elif os.path.exists(DEFAULTS_BOOK):
            raw = load_sample()
            source_label = "PREVIEW - sample day from Banglore Constraint1.xlsx, not today's orders"
    except Exception as exc:
        st.sidebar.error(f"Couldn't load today's customers: {exc}")

    # ── sidebar: solver ──
    st.sidebar.header("Solver")
    fewest = st.sidebar.toggle(
        "Use fewest vehicles first", value=True,
        help="On: minimise the number of vehicles, then cost. Off: pure cost - vehicles with "
             "₹0 fixed cost are then 'free', so nearby customers on different sides of an MDC "
             "may get separate bikes.")
    seconds = st.sidebar.slider("Time limit (seconds)", 10, 180, 30, 5,
                                help="Longer = better plans for big days.")

    st.title("🚚 MDC Route Planner")
    if raw is None:
        st.info("No customer data loaded - check the database connection in the sidebar.")
        st.stop()

    cust_all, notes, missing = prepare_customers(raw)
    if date_note:
        notes.insert(0, date_note)
    if missing:
        st.error(f"The customer data is missing column(s): {', '.join(missing)}. "
                 f"Columns found: {', '.join(map(str, raw.columns))}")
        st.stop()

    city_note = ""
    raw_n = normalise_columns(raw)
    if "CityId" in raw_n.columns and raw_n.CityId.nunique() > 1:
        cities = sorted(raw_n.CityId.dropna().unique().tolist())
        city = st.sidebar.selectbox("City", cities,
                                    index=cities.index(DEFAULT_CITY_ID) if DEFAULT_CITY_ID in cities else 0)
        keep_ids = set(raw_n.loc[raw_n.CityId == city, "CustomerId"])
        cust_all = cust_all[cust_all.CustomerId.isin(keep_ids)].reset_index(drop=True)
        city_note = f" · CityId {city}"

    st.caption(f"Customers: **{source_label}**{city_note} · loaded at {ss.fetched_at[1]} · "
               f"{len(cust_all)} customers · {cust_all.Tonnage.sum():,.1f} kg")
    if notes:
        with st.expander(f"⚠️ Data checks ({len(notes)})"):
            for msg, rows in notes:
                st.write(f"**{msg}** — {len(rows)} row(s)")
                st.dataframe(rows, width="stretch")
    if cust_all.empty:
        st.warning(f"No orders for {today:%d %b %Y} yet. Press **Refresh data** once they're in.")
        st.stop()

    # ── 1. customers ──
    st.subheader("1 · Customers and delivery windows")
    base = cust_all.copy()
    base.insert(0, "Include", [ss.incl_over.get(c, True) for c in base.CustomerId])
    src_win = dict(zip(base.CustomerId, base.DeliveryWindow))
    base["DeliveryWindow"] = [ss.win_over.get(c, w) for c, w in zip(base.CustomerId, base.DeliveryWindow)]

    c1, c2, _ = st.columns([2, 1, 3])
    bulk = c1.text_input("Set one window for every customer", placeholder="Same window for all, e.g. 9:30-13:30",
                         label_visibility="collapsed")
    if c2.button("Apply to all", disabled=not bulk):
        if window_ok(bulk):
            ss.win_over = {c: bulk.strip() for c in base.CustomerId}
            ss.cust_ver += 1
            st.rerun()
        else:
            st.error("Use HH:MM-HH:MM, e.g. 9:30-13:30")

    edited = st.data_editor(
        base, key=f"cust_{ss.cust_ver}_{signature(cust_all)}", hide_index=True, width="stretch",
        height=min(420, 38 + 35 * len(base)),
        disabled=["CustomerId", "Customer", "Latitude", "Longitude", "Tonnage"],
        column_config={
            "Include": st.column_config.CheckboxColumn(width="small"),
            "Tonnage": st.column_config.NumberColumn("Tonnage (kg)", format="%.2f"),
            "DeliveryWindow": st.column_config.TextColumn("Delivery window", help="HH:MM-HH:MM"),
            "Latitude": st.column_config.NumberColumn(format="%.6f"),
            "Longitude": st.column_config.NumberColumn(format="%.6f"),
        })
    for cid, inc, w in zip(edited.CustomerId, edited.Include, edited.DeliveryWindow):
        ss.incl_over[cid] = bool(inc)
        if str(w).strip() != src_win.get(cid):
            ss.win_over[cid] = str(w).strip()
        else:
            ss.win_over.pop(cid, None)

    cust = edited[edited.Include].drop(columns="Include").reset_index(drop=True)
    bad_win = cust[~cust.DeliveryWindow.apply(window_ok)]

    # ── 2 + 3. MDCs and vehicles ──
    mdc_default, fleet_default, defaults_src = default_tables()
    mdc_start = ss.get("cfg_mdcs", mdc_default)
    fleet_start = ss.get("cfg_fleet", fleet_default)
    cfg_src = ss.get("cfg_src", f"defaults from {defaults_src}")

    st.subheader("2 · MDCs and vehicles")
    st.caption(f"Using **{cfg_src}**. Edit below, or download → change in Excel → upload back. "
               f"One row per vehicle; untick Active to leave a vehicle out. "
               f"Distance is one-way (MDC → last customer).")

    left, right = st.columns([2, 5])
    with left:
        st.markdown("**MDCs**")
        mdcs = st.data_editor(
            mdc_start, key=f"mdc_{ss.cfg_ver}", num_rows="dynamic", hide_index=True, width="stretch",
            column_config={"Latitude": st.column_config.NumberColumn(format="%.6f"),
                           "Longitude": st.column_config.NumberColumn(format="%.6f")})
    with right:
        codes = [str(c).strip().upper() for c in mdcs.MDC.dropna() if str(c).strip()]
        n_active = int(fleet_start.Active.map(as_bool).sum())
        st.markdown(f"**Vehicles** ({len(fleet_start)} on file)")
        fleet = st.data_editor(
            fleet_start, key=f"fleet_{ss.cfg_ver}", num_rows="dynamic", hide_index=True, width="stretch",
            height=min(460, 38 + 35 * (len(fleet_start) + 1)),
            column_config={
                "Active": st.column_config.CheckboxColumn(width="small", default=True),
                "MDC": st.column_config.SelectboxColumn(options=codes or ["MDC1"], required=True),
                "Vehicle": st.column_config.TextColumn(required=True, help="e.g. Bike21 - unique per MDC"),
                "DropPoints": st.column_config.NumberColumn("Max drops", min_value=1, step=1),
                "MaxDistance_km": st.column_config.NumberColumn("Max one-way km", min_value=0.0),
                "MaxTonnage_kg": st.column_config.NumberColumn("Max kg", min_value=0.0),
                "FixedCost_Rs": st.column_config.NumberColumn("Fixed ₹", min_value=0.0, default=0.0),
                "VariableCost_Rs_per_km": st.column_config.NumberColumn("₹ / km", min_value=0.0, default=0.0),
                "ShiftStart": st.column_config.TextColumn("Shift start", help="HH:MM, blank = any"),
                "ShiftEnd": st.column_config.TextColumn("Shift end", help="HH:MM, blank = any"),
            })

    d1, d2, d3 = st.columns([1, 2, 1])
    d1.download_button("⬇️ Download MDCs + vehicles (.xlsx)", config_workbook(mdcs, fleet),
                       f"MDC_Vehicle_Config_{today:%Y%m%d}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    up = d2.file_uploader("⬆️ Upload updated MDCs + vehicles", type=["xlsx"], key=f"cfg_up_{ss.cfg_ver}",
                          label_visibility="collapsed")
    if d3.button("↺ Reset to defaults", width="stretch"):
        for k in ("cfg_mdcs", "cfg_fleet", "cfg_src"):
            ss.pop(k, None)
        ss.cfg_ver += 1
        st.rerun()
    if up is not None:
        new_mdcs, new_fleet, up_err = read_config_upload(up.getvalue())
        if up_err:
            for e in up_err:
                st.error(f"Upload not applied - {e}")
        else:
            ss.cfg_mdcs, ss.cfg_fleet = new_mdcs, new_fleet
            ss.cfg_src = f"uploaded file '{up.name}' ({datetime.now():%H:%M})"
            ss.cfg_ver += 1           # resets the editors (and the uploader) to the new data
            st.rerun()

    cfg, cfg_errors = build_config(mdcs, fleet)

    # ── run ──
    problems = list(cfg_errors)
    if len(bad_win):
        problems.append("Fix these delivery windows (HH:MM-HH:MM, start before end): " +
                        ", ".join(f"{r.Customer} ('{r.DeliveryWindow}')" for r in bad_win.itertuples()))
    if cust.empty:
        problems.append("No customers selected")
    for p in problems:
        st.error(p)
    if cfg and not cust.empty:
        heaviest = max(f[4] for f in cfg["fleet"])
        heavy = cust[cust.Tonnage > heaviest]
        if len(heavy):
            st.warning(f"{len(heavy)} customer(s) weigh more than any vehicle can carry "
                       f"({heaviest:g} kg) and will be unserved: " + ", ".join(heavy.Customer))

    inputs_sig = signature(cust, mdcs, fleet, fewest)
    n_veh = len(cfg["fleet"]) if cfg else 0
    if st.button(f"▶ Plan routes ({len(cust)} customers, {n_veh} active vehicles)", type="primary",
                 disabled=bool(problems), width="stretch"):
        sim.SOLVER_SECONDS = seconds
        with st.spinner(f"Planning {len(cust)} customers on OpenStreetMap roads "
                        f"(up to {seconds}s)…"):
            try:
                res = sim.solve_case("Plan", cfg, cust,
                                     vehicle_penalty_rs=VEHICLE_PENALTY_RS if fewest else 0.0)
                ss.plan = {"res": res, "cfg": cfg, "cust": cust, "mdcs": mdcs, "fleet": fleet,
                           "sig": inputs_sig, "fewest": fewest}
            except Exception as exc:
                st.error(f"Planning failed: {exc}")

    if "plan" not in ss:
        return
    plan = ss.plan
    res, pcfg, pcust = plan["res"], plan["cfg"], plan["cust"]
    if plan["sig"] != inputs_sig:
        st.info("Inputs changed since this plan was made - press **Plan routes** to update it.")

    # ── 4. results ──
    st.header("Plan")
    s = res["summary"]
    served = int(s.Stops.sum()) if len(s) else 0
    cost = s.TotalCost.sum() if len(s) else 0.0
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Vehicles used", f"{len(s)} / {res['fleet_size']}")
    m2.metric("Customers served", f"{served} / {len(pcust)}")
    m3.metric("One-way km", f"{s.OneWayDistance_km.sum():,.1f}" if len(s) else "0")
    m4.metric("Total cost", f"₹{cost:,.0f}")
    m5.metric("Cost / customer", f"₹{cost / served:,.0f}" if served else "-")
    st.caption(f"Distances: {res['source']} · "
               f"{'fewest vehicles first, then cost' if plan['fewest'] else 'lowest cost'}")

    if res["unserved"]:
        with st.expander(f"⚠️ {len(res['unserved'])} customer(s) not served - why", expanded=True):
            for idx in res["unserved"]:
                r = pcust.iloc[idx]
                st.write(f"**{r.Customer}** — {unserved_reason(r, pcfg)}")

    st.subheader("MDC split")
    st.dataframe(mdc_summary(res, pcfg), hide_index=True, width="stretch")

    st.subheader("Map")
    with st.spinner("Drawing routes along OpenStreetMap roads…"):
        fmap, fallbacks = build_map(res, pcfg, pcust)
    st_folium(fmap, height=620, use_container_width=True, returned_objects=[], key="plan_map")
    if fallbacks:
        st.caption(f"{fallbacks} route(s) drawn as dashed straight lines - the OpenStreetMap "
                   f"routing service didn't answer for them.")

    st.subheader("Customer → MDC → vehicle")
    st.dataframe(assignment_table(res, pcust), hide_index=True, width="stretch", height=380)
    with st.expander("Vehicle summary"):
        st.dataframe(s, hide_index=True, width="stretch")

    st.download_button(
        "⬇️ Download plan (.xlsx)", results_workbook(res, pcfg, pcust, plan["mdcs"], plan["fleet"]),
        f"MDC_Route_Plan_{today:%Y%m%d}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")


if __name__ == "__main__":
    main()

"""
app.py — Route Optimizer  |  Streamlit UI
Run: streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium

from optimizer import solve_vrp

# ──────────────────────────── Page Config ────────────────────────────

st.set_page_config(
    page_title="Route Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Custom CSS ─────────────────────────────

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 8px; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────── Header ─────────────────────────────────

st.title("🚚 Route Optimizer")
st.caption(
    "Cost-optimised delivery routing — time-window constraints · tonnage limits · max drop points"
)

# ──────────────────────────── Sidebar ────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Vehicle Parameters")
    num_vehicles = st.number_input(
        "Number of Vehicles", min_value=1, max_value=500, value=7
    )
    max_drops = st.number_input(
        "Max Drop Points / Vehicle", min_value=1, max_value=50, value=4
    )
    max_tonnage = st.number_input(
        "Max Tonnage / Vehicle (kg)", min_value=100, max_value=100_000, value=1500
    )
    max_crates = st.number_input(
        "Max Crates / Vehicle", min_value=1, max_value=500, value=90
    )
    speed_kmh = st.number_input(
        "Vehicle Speed (km/hr)", min_value=1, max_value=300, value=20
    )

    st.markdown("---")
    st.subheader("⏱️ Service Time")
    loading_sec = st.number_input(
        "Loading Time / Crate at FC (sec)", min_value=0, max_value=300, value=30
    )
    unloading_sec = st.number_input(
        "Unloading Time / Crate at Customer (sec)", min_value=0, max_value=300, value=45
    )
    buffer_min = st.number_input(
        "Buffer Time / Customer (min)", min_value=0, max_value=60, value=3
    )
    waiting_min = st.number_input(
        "Waiting Time / Customer (min)", min_value=0, max_value=60, value=5
    )
    max_svc_min = st.number_input(
        "Max Service Time / Customer (min)", min_value=1, max_value=120, value=45
    )

    st.markdown("---")
    st.subheader("Solver")
    use_ortools = st.toggle("Use OR-Tools (optimal)", value=True)
    if use_ortools:
        st.info(
            "Google OR-Tools — finds near-optimal routes. "
            "Time limit: 30 seconds. Falls back to greedy if unavailable."
        )
    else:
        st.info("Greedy nearest-neighbour — instant results, slightly sub-optimal.")

    st.markdown("---")
    st.subheader("Distance / Routing")
    routing_option = st.radio(
        "Distance Method",
        ["OSRM (Real Roads — Free)", "OpenRouteService (Real Roads — API Key)", "Haversine (Straight-Line)"],
        index=0,
    )

    ors_api_key = ""
    if routing_option.startswith("OSRM"):
        routing = "osrm"
        st.info("Uses OpenStreetMap road network via OSRM public server. No API key needed.")
    elif routing_option.startswith("OpenRoute"):
        routing = "ors"
        ors_api_key = st.text_input("OpenRouteService API Key", type="password",
                                    help="Get a free key at openrouteservice.org")
        if not ors_api_key:
            st.warning("Enter your ORS API key to use this option.")
    else:
        routing = "haversine"
        st.info("Straight-line (crow-flies) distances. Fastest, works offline.")

    st.markdown("---")
    st.subheader("Slot Overlap")
    allow_overlapping_slots = st.toggle("Allow Overlapping Slot Customers per Trip", value=True)
    if allow_overlapping_slots:
        st.info(
            "**Allowed (default)** — A route can have multiple customers from the "
            "same slot (e.g. two 5:00–6:00 customers on the same vehicle)."
        )
    else:
        st.warning(
            "**Not Allowed** — Each slot appears at most ONCE per route. "
            "E.g. two customers with 5:00–6:00 will always go on different vehicles. "
            "Different slots (5:00–6:00 + 5:30–6:30) can still share a route."
        )

    st.markdown("---")
    st.subheader("Trip Type")
    trip_type = st.radio(
        "Route Mode",
        ["🔄 Round Trip — Return to FC", "➡️ One-Way — End at Last Customer"],
        index=0,
    )
    round_trip = trip_type.startswith("🔄")
    if round_trip:
        st.info("Vehicle departs FC → delivers → returns to FC. Full loop distance counted.")
    else:
        st.info(
            "Vehicle departs FC → delivers → ends at last customer. "
            "Only outward distance counted (driver returns independently)."
        )

    st.markdown("---")
    st.markdown(
        "**Required CSV columns**\n"
        "- `Latitude`, `Longitude`\n"
        "- `Slot` (e.g. `5:00 AM - 6:00 AM`)\n"
        "- `FC Latitude`, `FC Longitude`\n"
        "- `Tonnage` or `OrderKg`\n"
        "- `TotalCrates` or `Crates`"
    )

# ────────────────────────── File Upload ──────────────────────────────

st.subheader("📁 Upload Orders CSV")
uploaded = st.file_uploader(
    "Drag & drop your orders CSV here",
    type=["csv"],
    label_visibility="collapsed",
)

REQUIRED_COLS = {"Latitude", "Longitude", "Slot", "FC Latitude", "FC Longitude"}

if not uploaded:
    st.info("Upload an orders CSV to get started.")
    st.stop()

# ──────────────────────── Load & Validate ────────────────────────────

df_raw = pd.read_csv(uploaded)
df_raw.columns = df_raw.columns.str.strip()

# Normalize weight column: accept Tonnage or OrderKg
if "Tonnage" not in df_raw.columns and "OrderKg" in df_raw.columns:
    df_raw = df_raw.rename(columns={"OrderKg": "Tonnage"})

missing_cols = REQUIRED_COLS - set(df_raw.columns)
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

if "Tonnage" not in df_raw.columns:
    st.error("Missing required column: `Tonnage` or `OrderKg`")
    st.stop()

df_raw = df_raw.dropna(subset=["Latitude", "Longitude", "Tonnage"])
df_raw = df_raw.reset_index(drop=True)

# ─────────────────────── Preview + Summary ───────────────────────────

col_data, col_stats = st.columns([3, 1])

with col_data:
    st.subheader(f"📋 Orders  ({len(df_raw)} rows)")
    st.dataframe(df_raw, use_container_width=True, height=280)

with col_stats:
    st.subheader("📊 Summary")
    total_tonnage = df_raw["Tonnage"].astype(float).sum()
    min_v_drops = int(np.ceil(len(df_raw) / max_drops))
    min_v_ton = int(np.ceil(total_tonnage / max_tonnage))
    min_v_needed = max(min_v_drops, min_v_ton)

    # Crate-based minimum vehicles
    crate_col_name = None
    if "TotalCrates" in df_raw.columns:
        crate_col_name = "TotalCrates"
    elif "Crates" in df_raw.columns:
        crate_col_name = "Crates"

    if crate_col_name and max_crates > 0:
        total_crates_all = df_raw[crate_col_name].astype(float).sum()
        min_v_crates = int(np.ceil(total_crates_all / max_crates))
        min_v_needed = max(min_v_needed, min_v_crates)
    else:
        total_crates_all = None
        min_v_crates = 0

    # Identify which constraint is the binding factor
    binding = "Drops"
    binding_val = min_v_drops
    if min_v_ton > binding_val:
        binding = "Tonnage"
        binding_val = min_v_ton
    if min_v_crates > binding_val:
        binding = "Crates"
        binding_val = min_v_crates

    st.metric("Total Orders", len(df_raw))
    st.metric("Total Tonnage (kg)", f"{total_tonnage:,.1f}")
    if total_crates_all is not None:
        st.metric("Total Crates", f"{total_crates_all:,.1f}")
    st.metric("Min Vehicles Needed", min_v_needed)
    st.caption(
        f"By drops: {min_v_drops} · By tonnage: {min_v_ton}"
        + (f" · By crates: {min_v_crates}" if min_v_crates else "")
        + f" — **limited by {binding}**"
    )

    if num_vehicles < min_v_needed:
        st.warning(f"⚠️ Need at least **{min_v_needed}** vehicles")
    else:
        st.success(f"✅ {num_vehicles} vehicles configured")

st.markdown("---")

# ── Pre-solve constraint warnings ────────────────────────────────────
_infeasible = []
_over_tonnage = df_raw[df_raw["Tonnage"].astype(float) > max_tonnage]
if not _over_tonnage.empty:
    for _, r in _over_tonnage.iterrows():
        _infeasible.append(f"**{r.get('Customer', r.name)}** — Tonnage {r['Tonnage']} kg exceeds max {max_tonnage} kg")

if crate_col_name and max_crates > 0:
    _over_crates = df_raw[df_raw[crate_col_name].astype(float) > max_crates]
    if not _over_crates.empty:
        for _, r in _over_crates.iterrows():
            _infeasible.append(f"**{r.get('Customer', r.name)}** — {r[crate_col_name]} crates exceeds max {max_crates} crates")

if _infeasible:
    st.warning(
        "⚠️ The following customers **cannot be served** as they individually exceed vehicle limits — "
        "they will be left unserved regardless of how many vehicles are used:\n\n"
        + "\n\n".join(f"- {m}" for m in _infeasible)
    )

# ──────────────────────── Optimize Button ────────────────────────────

if st.button("🚀 Optimize Routes", type="primary", use_container_width=True):
    spinner_msg = {
        "osrm": "Fetching real road distances from OSRM, then optimizing…",
        "ors": "Fetching real road distances from OpenRouteService, then optimizing…",
        "haversine": "Optimizing routes…",
    }.get(routing, "Optimizing routes…")

    with st.spinner(spinner_msg):
        result = solve_vrp(
            df_raw, num_vehicles, max_drops, max_tonnage, speed_kmh,
            use_ortools, routing, ors_api_key,
            allow_overlapping_slots=allow_overlapping_slots,
            round_trip=round_trip,
            max_crates=max_crates,
            loading_sec_per_crate=float(loading_sec),
            unloading_sec_per_crate=float(unloading_sec),
            buffer_time_min=float(buffer_min),
            waiting_time_min=float(waiting_min),
            max_service_time_min=float(max_svc_min),
        )
    if result is None or not result.get("routes"):
        st.error(
            "❌ No feasible solution found. "
            "Try increasing the number of vehicles or relaxing constraints."
        )
        st.stop()

    st.session_state["result"] = result
    st.session_state["df_orig"] = df_raw

# ──────────────────────────── Results ────────────────────────────────

if "result" not in st.session_state:
    st.stop()

result: dict = st.session_state["result"]
df_orig: pd.DataFrame = st.session_state["df_orig"]

unserved_str = (
    f" | ⚠️ **{result['unserved']} orders unserved**" if result["unserved"] else ""
)
routing_label = {
    "osrm": "OSRM (real roads)",
    "ors": "OpenRouteService (real roads)",
    "haversine": "Haversine (straight-line)",
}.get(result.get("routing_source", "haversine"), result.get("routing_source", ""))

slot_label = "Overlapping Slots Allowed" if allow_overlapping_slots else "No Overlapping Slots"
trip_label = "Round Trip" if round_trip else "One-Way"

st.success(
    f"✅ Optimization complete — "
    f"**{result['vehicles_used']} vehicles** | "
    f"**{result['total_distance']:.1f} km** | "
    f"{routing_label} | {slot_label} | {trip_label}"
    + unserved_str
)

tab_summary, tab_routes, tab_map = st.tabs(
    ["📊 Route Summary", "📋 Detailed Routes", "🗺️ Map"]
)

# ── Summary Tab ──────────────────────────────────────────────────────
with tab_summary:
    summary_df = result["summary_df"]
    st.dataframe(summary_df, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vehicles Used", result["vehicles_used"])
    c2.metric("Total Distance", f"{result['total_distance']:.1f} km")
    c3.metric("Total Orders", len(df_orig))
    c4.metric("Unserved Orders", result["unserved"])

    if "TotalCrates" in summary_df.columns:
        total_svc = summary_df["TotalServiceTime_min"].sum()
        cl1, cl2 = st.columns(2)
        cl1.metric("Total Crates", f"{summary_df['TotalCrates'].sum():.0f}")
        cl2.metric("Total Service Time", f"{total_svc:.1f} min")

    if not summary_df.empty:
        csv_summary = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Summary CSV",
            csv_summary,
            "route_summary.csv",
            "text/csv",
        )

# ── Detailed Routes Tab ──────────────────────────────────────────────
with tab_routes:
    route_df = result["route_df"]

    # Filter by vehicle
    vehicles = ["All"] + list(route_df["Vehicle"].unique()) if not route_df.empty else ["All"]
    selected_vehicle = st.selectbox("Filter by Vehicle", vehicles)

    filtered_df = route_df if selected_vehicle == "All" else route_df[route_df["Vehicle"] == selected_vehicle]

    # Build focused display: only the columns users care about
    display_cols = ["Vehicle", "StopNumber", "TotalStops"]
    for c in ["Customer", "Slot", "Tonnage", "Crates"]:
        if c in filtered_df.columns:
            display_cols.append(c)
    display_cols += [
        "EstimatedArrival",
        "LoadingTime_min",
        "UnloadingTime_min",
        "BufferTime_min",
        "WaitingTime_min",
        "OverallServiceTime_min",
        "TravelToNext_min",
        "TravelToNext_km",
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    # Rename for readability
    col_labels = {
        "StopNumber":            "Stop #",
        "TotalStops":            "Total Stops",
        "EstimatedArrival":      "Arrival Time",
        "LoadingTime_min":       "Loading (min)",
        "UnloadingTime_min":     "Unloading (min)",
        "BufferTime_min":        "Buffer (min)",
        "WaitingTime_min":       "Waiting (min)",
        "OverallServiceTime_min": "Total Service (min)",
        "TravelToNext_min":      "Travel to Next (min)",
        "TravelToNext_km":       "Travel to Next (km)",
    }

    display_df = filtered_df[display_cols].rename(columns=col_labels)
    st.dataframe(display_df, use_container_width=True, height=450)

    csv_routes = route_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Routes CSV",
        csv_routes,
        "optimized_routes.csv",
        "text/csv",
        use_container_width=True,
    )

# ── Map Tab ──────────────────────────────────────────────────────────
with tab_map:
    fc_lat = float(df_orig["FC Latitude"].iloc[0])
    fc_lon = float(df_orig["FC Longitude"].iloc[0])

    m = folium.Map(
        location=[fc_lat, fc_lon],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    # FC / MDC depot marker
    folium.Marker(
        [fc_lat, fc_lon],
        tooltip="FC / MDC (Depot)",
        popup=folium.Popup("<b>FC / MDC — Depot</b>", max_width=160),
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)

    COLORS = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#9A6324", "#469990", "#800000",
        "#808000", "#000075", "#a9a9a9", "#dcbeff", "#aaffc3",
        "#ffe119", "#4169e1", "#ff7f50", "#40e0d0", "#da70d6",
    ]

    for route_struct in result["routes"]:
        vid = route_struct["vehicle_id"]
        color = COLORS[vid % len(COLORS)]
        stops = route_struct["stops"]

        # Build polyline: depot → stops → depot
        coords = [[fc_lat, fc_lon]]
        for s in stops:
            coords.append([s["lat"], s["lon"]])

            # Numbered circle marker
            popup_html = (
                f"<div style='font-family:sans-serif;font-size:13px;min-width:220px;'>"
                f"<div style='background:{color};color:white;padding:6px 10px;border-radius:6px 6px 0 0;margin-bottom:6px;'>"
                f"<b>Vehicle {vid + 1} &nbsp;·&nbsp; Stop {s['stop_num']} of {len(stops)}</b>"
                f"</div>"
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<tr><td style='color:#555;padding:2px 4px;'>Customer ID</td>"
                f"    <td style='padding:2px 4px;'><b>{s['customer_id']}</b></td></tr>"
                f"<tr><td style='color:#555;padding:2px 4px;'>Name</td>"
                f"    <td style='padding:2px 4px;'><b>{s['customer']}</b></td></tr>"
                f"<tr><td style='color:#555;padding:2px 4px;'>Slot</td>"
                f"    <td style='padding:2px 4px;'>{s['slot']}</td></tr>"
                f"<tr><td style='color:#555;padding:2px 4px;'>Arrival</td>"
                f"    <td style='padding:2px 4px;'><b style='color:#1a7a1a;'>{s['arrival_str']}</b></td></tr>"
                f"<tr><td style='color:#555;padding:2px 4px;'>Tonnage</td>"
                f"    <td style='padding:2px 4px;'>{s['tonnage']} kg</td></tr>"
                f"<tr><td style='color:#555;padding:2px 4px;'>Crates</td>"
                f"    <td style='padding:2px 4px;'>{s['crates']:.0f}</td></tr>"
                f"<tr><td style='color:#555;padding:2px 4px;'>Service Time</td>"
                f"    <td style='padding:2px 4px;'>{s['overall_svc']} min</td></tr>"
                + (
                    f"<tr><td style='color:#555;padding:2px 4px;'>Travel to Next</td>"
                    f"    <td style='padding:2px 4px;'>{s['travel_to_next_min']} min &nbsp;·&nbsp; {s['travel_to_next_km']} km</td></tr>"
                    if s.get('travel_to_next_min') is not None else ""
                )
                + f"</table></div>"
            )
            folium.CircleMarker(
                [s["lat"], s["lon"]],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=(
                    f"<b>V{vid + 1} · Stop {s['stop_num']}</b> &nbsp;|&nbsp; "
                    f"{s['customer_id']} · {s['customer'][:28]}<br>"
                    f"Slot: {s['slot']} &nbsp;·&nbsp; ETA: {s['arrival_str']}"
                ),
            ).add_to(m)

            folium.Marker(
                [s["lat"], s["lon"]],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="'
                        f"font-size:10px;font-weight:bold;color:white;"
                        f"background:{color};border-radius:50%;"
                        f"width:20px;height:20px;text-align:center;"
                        f'line-height:20px;">'
                        f"{s['stop_num']}</div>"
                    ),
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                ),
            ).add_to(m)

        is_round_trip = route_struct.get("round_trip", True)
        if is_round_trip:
            coords.append([fc_lat, fc_lon])   # close the loop back to FC

        # Solid line = round trip; dashed line = one-way
        folium.PolyLine(
            coords,
            color=color,
            weight=3,
            opacity=0.85,
            dash_array=None if is_round_trip else "8 6",
            tooltip=(
                f"Vehicle {vid + 1} | "
                f"{len(stops)} stops | "
                f"{route_struct['total_distance']:.1f} km | "
                f"{'Round Trip' if is_round_trip else 'One-Way'}"
            ),
        ).add_to(m)

    # ── Map legend ───────────────────────────────────────────────────
    legend_items = "".join(
        f'<div style="margin-bottom:4px;">'
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'border-radius:50%;background:{COLORS[r["vehicle_id"] % len(COLORS)]};'
        f'margin-right:6px;vertical-align:middle;"></span>'
        f'<b>Vehicle {r["vehicle_id"] + 1}</b> &nbsp;'
        f'{len(r["stops"])} stops · {r["total_distance"]:.1f} km</div>'
        for r in result["routes"]
    )
    legend_html = (
        '<div style="position:fixed;bottom:40px;left:40px;z-index:9999;'
        "background:white;padding:12px 16px;border-radius:10px;"
        'border:1px solid #ccc;font-size:13px;max-height:300px;overflow-y:auto;">'
        f"<b>🚚 Vehicles</b><br><br>{legend_items}</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, use_container_width=True, height=620)

# PHD Route Optimizer

Daily delivery route planner for Bengaluru's MDCs. It loads today's customer
orders from the `PHD_MDCRO_Base` table, assigns every customer to an MDC and
a vehicle, and plans one-way delivery routes (MDC → last customer) at the
lowest cost, shown on an OpenStreetMap map.

## Features

- **Today's orders only:** customers and tonnage come from `PHD_MDCRO_Base`,
  filtered to today. **Refresh data** picks up new orders.
- **Editable MDCs and vehicles:** one row per vehicle with its maximum drops,
  one-way distance cap, capacity in kg, fixed cost, cost per km and optional
  shift times. Download the settings as Excel, edit them, and upload the file
  back. Defaults come from `default_config.xlsx` (MDC1 + MDC2 Boomanalli, 40
  vehicles).
- **Editable delivery windows** per customer, or one window applied to all.
- **Solver:** Google OR-Tools on real road distances from OSRM, which uses
  OpenStreetMap data, so no API key is needed. By default it uses the fewest
  vehicles first and then the lowest cost. Customers that can't be served are
  listed with the reason.
- **Map:** OpenStreetMap tiles, with routes drawn along the actual roads.
- **Data checks:** bad coordinates, placeholder locations and orders from other
  days are excluded and listed, never silently dropped.
- **Download:** the plan as Excel, with the customer → MDC → vehicle
  assignment, routes and costs.

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> → **Create app** → **Deploy a public app
   from GitHub**.
2. Repository `Manojkumarninja/PHD-RouteOptimizer`, branch `main`,
   main file path **`streamlit_app.py`** (`app.py` and `mdc_streamlit_app.py` open the same app).
3. Under **Advanced settings → Secrets**, paste your database connection:

   ```toml
   [connections.phd_db]
   dialect = "mysql"
   driver = "pymysql"
   host = "YOUR_DB_HOST"
   port = 3306
   username = "YOUR_USER"
   password = "YOUR_PASSWORD"
   database = "YOUR_DATABASE"

   [phd_mdcro]
   query = "SELECT * FROM PHD_MDCRO_Base WHERE DATE(DeliveryDate) = :today"
   ```

4. **Deploy.** The database must accept connections from Streamlit Cloud. If
   it only allows whitelisted IPs, ask your DB admin to allow Streamlit
   Cloud's outbound IPs.

Never commit real credentials: `.streamlit/secrets.toml` is git-ignored, and
this repository is public.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill it in
streamlit run mdc_streamlit_app.py
```

## Files

| File | What it is |
|---|---|
| `mdc_streamlit_app.py` | The MDC route planner app |
| `simulate_bangalore.py` | Multi-depot routing engine (OR-Tools); also a command-line tool for comparing scenarios from a constraints workbook |
| `default_config.xlsx` | Default MDCs + vehicles. Replace it to change the defaults everyone sees |
| `app.py`, `streamlit_app.py` | Entry points - both open the MDC route planner, so either works as the Streamlit Cloud main file |
| `single_depot_app.py`, `optimizer.py` | Original single-depot route planner (`streamlit run single_depot_app.py`); the engine reuses its distance helpers |
| `plot_simulation.py` | Route-map PNGs for scenario comparisons |

## Input table: `PHD_MDCRO_Base`

| Column | Used as |
|---|---|
| `DeliveryDate` | Only today's rows are used |
| `CustomerId`, `Customer` | Customer; several orders for the same customer are merged and their tonnage summed |
| `Latitude`, `Longitude` | Delivery location |
| `Tonnage` | Load in kg |
| `DeliveryWindow` *(optional)* | e.g. `9:30-13:30`; defaults to 9:30-13:30 and can be edited in the app |

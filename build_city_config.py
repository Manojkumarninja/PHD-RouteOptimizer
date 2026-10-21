"""
build_city_config.py - save a city's starter MDCs + vehicles as
default_config_<city>.xlsx (committed with the app, so everyone gets the same
defaults). The app already builds these on the fly for any new
PHD_MDCRO_Base_XXX table; use this to freeze them, then edit as needed.

    python build_city_config.py Delhi PHD_MDCRO_Base_DLH
    python build_city_config.py Mumbai PHD_MDCRO_Base_MUM --mdcs 2

Reads customers from the database (connection in .streamlit/secrets.toml) and
cleans them the same way the app does. See city_config.py for the method.
"""
import argparse
import os
import sys
import tomllib

import pandas as pd
import sqlalchemy as sa

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
import mdc_streamlit_app as app  # noqa: E402  (prepare_customers, config_workbook)
from city_config import build_starter  # noqa: E402


def load_customers(table: str) -> pd.DataFrame:
    if not app.TABLE_RE.fullmatch(table):
        raise SystemExit(f"'{table}' doesn't look like a PHD_MDCRO_Base table name")
    c = tomllib.load(open(os.path.join(APP_DIR, ".streamlit", "secrets.toml"), "rb"))["connections"]["phd_db"]
    eng = sa.create_engine(sa.engine.URL.create(
        f"{c.get('dialect', 'mysql')}+{c.get('driver', 'pymysql')}", username=c["username"],
        password=c["password"], host=c["host"], port=c["port"], database=c["database"]))
    raw = pd.read_sql(sa.text(f"SELECT * FROM {table}"), eng)
    cust, notes, missing = app.prepare_customers(raw)
    if missing:
        raise SystemExit(f"{table} is missing columns: {missing}")
    for msg, rows in notes:
        print(f"  note: {msg} ({len(rows)})")
    return cust


def build(city: str, table: str, n_mdcs: int = 2):
    cust = load_customers(table)
    print(f"{city}: {len(cust)} customers from {table}")
    mdcs, vehicles, log = build_starter(cust, n_mdcs)
    for line in log:
        print("  " + line)
    out = os.path.join(APP_DIR, f"default_config_{city.lower()}.xlsx")
    with open(out, "wb") as fh:
        fh.write(app.config_workbook(mdcs, vehicles))
    print(f"  wrote {os.path.basename(out)}  (whitelist it in .gitignore to commit it)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("city")
    ap.add_argument("table")
    ap.add_argument("--mdcs", type=int, default=2)
    a = ap.parse_args()
    build(a.city, a.table, a.mdcs)

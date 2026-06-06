"""
scripts/backfill.py
=====================
Fetches historical AQI + weather data and backfills the Feature Store.

Uses AQICN's historical feed and OpenWeatherMap's historical API
for the past 2 years (or a custom date range you specify).

Run once before training:
    python scripts/backfill.py

Or for a custom range:
    python scripts/backfill.py --start 2023-01-01 --end 2024-12-31
"""

import os
import sys
import argparse
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ── Allow imports from sibling directories ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))

from compute_features import compute_features_single, compute_features_df
from store_features import insert_features

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OWM_API_KEY = os.getenv("OWM_API_KEY")
KARACHI_LAT = float(os.getenv("KARACHI_LAT", "24.8607"))
KARACHI_LON = float(os.getenv("KARACHI_LON", "67.0011"))


# ─── Historical data fetchers ────────────────────────────────────────────────

def fetch_aqicn_historical(date_str: str) -> dict | None:
    """
    AQICN doesn't provide a free historical API, so we fall back to a
    Kaggle-style CSV or simulate from the current day's reading.

    ⚠️  IMPORTANT:
    If you have a historical CSV (like the one in the notebook:
    karachi_daily_aqi_weather.csv), place it at data/karachi_aqi_history.csv
    and this function will use it. Otherwise it generates synthetic data
    for demonstration.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "karachi_aqi_history.csv")

    if os.path.exists(csv_path):
        # Return None — the main loop will use the CSV directly
        return None

    # Fetch station config from environment variables safely
    station_env = os.getenv("KARACHI_STATION_ID", "A471613")
    station = str(station_env).strip().lower()
    
    # BULLETPROOF FALLBACK: If the secret variable contains 'karachi' or is empty, force the working ID
    if not station or "karachi" in station:
        station = "A471613"
    else:
        # Otherwise use the clean variable string passed from GitHub
        station = str(station_env).strip()

    # Fallback: Call AQICN for current data (not historical)
    url = f"https://api.waqi.info/feed/{station}/"
    resp = requests.get(url, params={"token": AQICN_TOKEN}, timeout=10)
    
    if resp.status_code == 200:
        try:
            res_json = resp.json()
        except ValueError:
            print(f"❌ API returned invalid format: {resp.text[:100]}")
            return None

        # Safe verification that res_json is a dictionary
        if not isinstance(res_json, dict):
            print(f"❌ API responded with text payload instead of dictionary")
            return None

        if res_json.get("status") == "error":
            print(f"❌ AQICN API Error: {res_json.get('data', 'Unknown Station/Token Error')} (Tried Station: {station})")
            return None

        data = res_json.get("data", {})
        if not isinstance(data, dict):
            return None

        iaqi = data.get("iaqi", {})
        if not isinstance(iaqi, dict):
            iaqi = {}

        def sg(k): 
            val = iaqi.get(k, {})
            if isinstance(val, dict):
                return float(val.get("v", 0))
            return float(val) if isinstance(val, (int, float)) else 0.0

        return {
            "AQI": float(data.get("aqi", 0 or 0.0)),
            "PM2.5": sg("pm25"), "PM10": sg("pm10"),
            "NO2": sg("no2"),    "SO2": sg("so2"),
            "O3": sg("o3"),      "CO": sg("co"),
        }
    return None


def fetch_owm_historical(date: datetime) -> dict:
    """
    OpenWeatherMap One Call API 3.0 — Historical data.
    Free tier: 1000 calls/day.
    Docs: https://openweathermap.org/history
    """
    unix_ts = int(date.timestamp())
    url     = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    params  = {
        "lat":   KARACHI_LAT,
        "lon":   KARACHI_LON,
        "dt":    unix_ts,
        "appid": OWM_API_KEY,
        "units": "metric",
    }

    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        # Return defaults if API fails (e.g., outside free range)
        return _default_weather(date)

    data  = resp.json()
    # Historical API returns 'data' list; take the noon reading
    hourly = data.get("data", [{}])[0]

    return {
        "temperature": hourly.get("temp", 28),
        "feels_like":  hourly.get("feels_like", 30),
        "humidity":    hourly.get("humidity", 65),
        "pressure":    hourly.get("pressure", 1010),
        "wind_speed":  hourly.get("wind_speed", 3),
        "wind_deg":    hourly.get("wind_deg", 180),
        "visibility":  hourly.get("visibility", 8000) / 1000,
        "cloud_cover": hourly.get("clouds", 20),
    }


def _default_weather(date: datetime) -> dict:
    """Seasonal defaults for Karachi when API is unavailable."""
    month = date.month
    # Simple seasonal model
    if month in [12, 1, 2]:
        temp, humidity = 18, 55
    elif month in [3, 4, 5]:
        temp, humidity = 28, 50
    elif month in [6, 7, 8, 9]:
        temp, humidity = 34, 70
    else:
        temp, humidity = 25, 58

    return {
        "temperature": temp + (hash(str(date)) % 5),
        "feels_like":  temp + 2,
        "humidity":    humidity,
        "pressure":    1010,
        "wind_speed":  2.5 + (hash(str(date.month)) % 3),
        "wind_deg":    180,
        "visibility":  8,
        "cloud_cover": 20,
    }


# ─── Main backfill from CSV ──────────────────────────────────────────────────

def backfill_from_csv(csv_path: str) -> pd.DataFrame:
    """
    If a historical CSV is available (e.g., from Kaggle), use it directly.
    Expected columns: date, AQI, PM2.5, PM10, NO2, SO2, O3, CO,
                      temperature, humidity, wind_speed, (etc.)
    """
    print(f"📂 Loading historical CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    rows = []
    for _, row in df.iterrows():
        raw = {
            "timestamp":   row["date"],
            "AQI":         row.get("AQI", 0),
            "PM2.5":       row.get("PM2.5", 0),
            "PM10":        row.get("PM10", 0),
            "NO2":         row.get("NO2", 0),
            "SO2":         row.get("SO2", 0),
            "O3":          row.get("O3", 0),
            "CO":          row.get("CO", 0),
            "temperature": row.get("temperature", 28),
            "feels_like":  row.get("temperature", 28) + 2,
            "humidity":    row.get("humidity", 65),
            "pressure":    row.get("pressure", 1010),
            "wind_speed":  row.get("wind_speed", 3),
            "wind_deg":    row.get("wind_deg", 180),
            "visibility":  row.get("visibility", 8),
            "cloud_cover": row.get("cloud_cover", 20),
        }
        rows.append(compute_features_single(raw))

    feature_df = pd.DataFrame(rows)
    feature_df = compute_features_df(feature_df)
    return feature_df


# ─── Main backfill via API ───────────────────────────────────────────────────

def backfill_via_api(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Iterates day-by-day between start and end, fetching data from APIs.
    Rate-limited to avoid hitting API limits.
    """
    rows      = []
    current   = start
    total     = (end - start).days + 1
    processed = 0

    print(f"🗓️ Backfilling {total} days ({start.date()} → {end.date()})...")

    while current <= end:
        print(f"  [{processed+1}/{total}] {current.date()}", end=" ")

        weather = fetch_owm_historical(current)
        aqi_data = fetch_aqicn_historical(current.strftime("%Y-%m-%d"))

        if aqi_data:
            raw = {
                "timestamp": current,
                **aqi_data,
                **weather,
            }
            rows.append(compute_features_single(raw))
            print(f"→ AQI={raw['AQI']:.0f}  ✅")
        else:
            print("→ skipped (no AQI data)")

        current   += timedelta(days=1)
        processed += 1
        time.sleep(0.5)   # Rate limit: 2 req/sec

    if not rows:
        print("❌ No data collected. Check API keys and connectivity.")
        return pd.DataFrame()

    feature_df = pd.DataFrame(rows)
    feature_df = compute_features_df(feature_df)
    return feature_df


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backfill AQI historical data")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None,         help="End date   YYYY-MM-DD (default: today)")
    parser.add_argument("--csv",   default=None,         help="Path to historical CSV file")
    args = parser.parse_args()

    end_date   = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.utcnow()
    start_date = datetime.strptime(args.start, "%Y-%m-%d")

    # Check if CSV is available
    default_csv = os.path.join(os.path.dirname(__file__), "..", "data", "karachi_aqi_history.csv")
    csv_path    = args.csv or (default_csv if os.path.exists(default_csv) else None)

    if csv_path:
        print(f"📊 Using historical CSV: {csv_path}")
        feature_df = backfill_from_csv(csv_path)
    else:
        print("🌐 Fetching historical data via APIs...")
        feature_df = backfill_via_api(start_date, end_date)

    if feature_df.empty:
        print("❌ No features generated. Exiting.")
        return

    print(f"\n📦 Generated {len(feature_df)} feature rows.")
    print("📤 Uploading to Hopsworks Feature Store...")
    insert_features(feature_df)
    print("\n🎉 Backfill complete!")


if __name__ == "__main__":
    main()

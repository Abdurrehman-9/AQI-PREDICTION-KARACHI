"""
scripts/backfill.py
=====================
Backfills the Hopsworks Feature Store with historical AQI data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))

from compute_features import compute_features_single, compute_features_df
from store_features import insert_features

load_dotenv()

OWM_API_KEY = os.getenv("OWM_API_KEY")
KARACHI_LAT = float(os.getenv("KARACHI_LAT", "24.8607"))
KARACHI_LON = float(os.getenv("KARACHI_LON", "67.0011"))

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "karachi_daily_aqi_weather.csv"
)

CSV_END_DATE = datetime(2025, 8, 5)
np.random.seed(42)

def load_kaggle_csv(csv_path: str) -> pd.DataFrame:
    print(f"  Loading Kaggle CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        raw = {
            "timestamp":   row["date"],
            "AQI":         float(row.get("AQI",         100)),
            "PM2.5":       float(row.get("PM2.5",         0)),
            "PM10":        float(row.get("PM10",           0)),
            "NO2":         float(row.get("NO2",            0)),
            "SO2":         float(row.get("SO2",            0)),
            "O3":          float(row.get("O3",             0)),
            "CO":          float(row.get("CO",             0)),
            "temperature": float(row.get("Temperature",  28)),
            "feels_like":  float(row.get("Temperature",  28)) + 2,
            "humidity":    float(row.get("Humidity",      65)),
            "pressure":    1010.0,
            "wind_speed":  _season_wind(row["date"].month),
            "wind_deg":    180.0,
            "visibility":  8.0,
            "cloud_cover": 20.0,
        }
        rows.append(compute_features_single(raw))

    feature_df = pd.DataFrame(rows)
    feature_df = compute_features_df(feature_df)
    return feature_df


def _season_wind(month: int) -> float:
    if month in [6, 7, 8, 9]: return 5.5
    elif month in [3, 4, 5]: return 4.0
    elif month in [12, 1, 2]: return 3.0
    else: return 3.5


MONTHLY_AQI = {
    1: 175, 2: 165, 3: 145, 4: 130, 5: 125,
    6: 110, 7:  90, 8:  85, 9:  95, 10: 120,
    11: 150, 12: 170,
}
DOW_ADJ = {0: +8, 1: +8, 2: +6, 3: +6, 4: +10, 5: +2, 6: 0}


def _synthetic_aqi(date: datetime) -> dict:
    base  = MONTHLY_AQI[date.month]
    dow   = DOW_ADJ[date.weekday()]
    noise = np.random.normal(0, 15)
    aqi   = float(np.clip(base + dow + noise, 30, 300))
    return {
        "AQI":   round(aqi, 1),
        "PM2.5": round(max(0, aqi * 0.38 + np.random.normal(0, 3)), 1),
        "PM10":  round(max(0, aqi * 0.65 + np.random.normal(0, 5)), 1),
        "NO2":   round(max(0, aqi * 0.14 + np.random.normal(0, 2)), 1),
        "SO2":   round(max(0, aqi * 0.05 + np.random.normal(0, 1)), 1),
        "O3":    round(max(0, aqi * 0.30 + np.random.normal(0, 4)), 1),
        "CO":    round(max(0, aqi * 2.80 + np.random.normal(0, 20)), 1),
    }


def _seasonal_weather(date: datetime) -> dict:
    month = date.month
    if month in [12, 1, 2]: temp, hum, wind = 18, 55, 3.0
    elif month in [3, 4, 5]: temp, hum, wind = 30, 48, 4.0
    elif month in [6, 7, 8, 9]: temp, hum, wind = 33, 75, 5.5
    else: temp, hum, wind = 26, 58, 3.5

    rng  = np.random.RandomState(int(date.strftime("%Y%m%d")))
    temp = round(temp + rng.uniform(-3, 3), 1)
    return {
        "temperature": temp,
        "feels_like":  round(temp + 2, 1),
        "humidity":    int(hum + rng.randint(-5, 5)),
        "pressure":    int(1010 + rng.randint(-3, 3)),
        "wind_speed":  round(wind + rng.uniform(-1, 1), 1),
        "wind_deg":    int(180 + rng.randint(-30, 30)),
        "visibility":  round(8 + rng.uniform(-2, 2), 1),
        "cloud_cover": int(rng.randint(10, 40)),
    }


def generate_synthetic_gap(start: datetime, end: datetime) -> pd.DataFrame:
    if start > end: return pd.DataFrame()
    rows = []
    current = start
    while current <= end:
        raw = {
            "timestamp": current,
            **_synthetic_aqi(current),
            **_seasonal_weather(current),
        }
        rows.append(compute_features_single(raw))
        current += timedelta(days=1)
    return compute_features_df(pd.DataFrame(rows))


def run_backfill(csv_path: str, synthetic_start: datetime, synthetic_end: datetime):
    all_frames = []

    if os.path.exists(csv_path):
        all_frames.append(load_kaggle_csv(csv_path))

    gap_df = generate_synthetic_gap(synthetic_start, synthetic_end)
    if not gap_df.empty:
        all_frames.append(gap_df)

    if not all_frames: return

    combined = pd.concat(all_frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    insert_features(combined)


def main():
    parser = argparse.ArgumentParser(description="3-layer AQI backfill")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--gap-start", default=None)
    parser.add_argument("--gap-end", default=None)
    args = parser.parse_args()

    gap_start = datetime.strptime(args.gap_start, "%Y-%m-%d") if args.gap_start else CSV_END_DATE + timedelta(days=1)
    gap_end = datetime.strptime(args.gap_end, "%Y-%m-%d") if args.gap_end else datetime.now() - timedelta(days=1)

    run_backfill(args.csv, gap_start, gap_end)


if __name__ == "__main__":
    main()

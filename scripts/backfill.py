"""
scripts/backfill.py
=====================
Backfills the Hopsworks Feature Store with historical AQI data
using a 3-layer strategy:

  LAYER 1 — Kaggle CSV (2023-01-01 to 2025-08-05)
            Real, verified Karachi AQI + weather data
            Most trustworthy source for model training
            Source: https://www.kaggle.com/datasets/sheemamasood/karachi-daily-aqi-weather

  LAYER 2 — Synthetic (2025-08-06 to yesterday)
            Fills the gap between CSV end date and today
            Uses Karachi's documented seasonal pollution patterns
            Only used where real data does not exist

  LAYER 3 — Live AQICN API (today onwards, every hour)
            Real data flowing in automatically via the hourly
            feature pipeline — no action needed here

WHY NOT USE AQICN FOR HISTORICAL DATA:
  AQICN provides NO free historical API. Their endpoint only
  returns the current live reading regardless of date passed.
  Calling it for past dates returns errors or today's value.
  This is a known limitation of their free tier.

Run once before training:
    python scripts/backfill.py

Custom range (overrides the CSV gap-fill dates):
    python scripts/backfill.py --start 2023-01-01 --end 2025-08-05
"""

import os
import sys
import argparse
import time
import requests
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

# Default CSV path — place your Kaggle file here
CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "karachi_daily_aqi_weather.csv"
)

# The Kaggle CSV covers up to this date
# Synthetic data fills the gap from this date to today
CSV_END_DATE = datetime(2025, 8, 5)

np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — KAGGLE CSV LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_kaggle_csv(csv_path: str) -> pd.DataFrame:
    """
    Loads the Kaggle CSV:
    kaggle.com/datasets/sheemamasood/karachi-daily-aqi-weather

    Columns expected:
      date, AQI, PM2.5, PM10, NO2, SO2, CO, O3,
      Temperature, Humidity, Precipitation, Next_Day_AQI

    Returns a feature-engineered DataFrame ready to insert
    into the Hopsworks Feature Store.
    """
    print(f"  Loading Kaggle CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalise column names — the CSV uses Title Case
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  CSV loaded: {len(df)} rows  "
          f"({df['date'].min().date()} to {df['date'].max().date()})")

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
            # CSV uses 'Temperature' (capital T)
            "temperature": float(row.get("Temperature",  28)),
            "feels_like":  float(row.get("Temperature",  28)) + 2,
            "humidity":    float(row.get("Humidity",      65)),
            # CSV has no pressure/wind/visibility — use seasonal defaults
            "pressure":    1010.0,
            "wind_speed":  _season_wind(row["date"].month),
            "wind_deg":    180.0,
            "visibility":  8.0,
            "cloud_cover": 20.0,
        }
        rows.append(compute_features_single(raw))

    feature_df = pd.DataFrame(rows)
    feature_df = compute_features_df(feature_df)
    print(f"  CSV features engineered: {len(feature_df)} rows after lag computation")
    return feature_df


def _season_wind(month: int) -> float:
    """Karachi seasonal wind speed defaults (m/s)."""
    if month in [6, 7, 8, 9]:   # monsoon
        return 5.5
    elif month in [3, 4, 5]:     # spring
        return 4.0
    elif month in [12, 1, 2]:    # winter
        return 3.0
    else:
        return 3.5


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — SYNTHETIC GAP FILLER
# Covers 2025-08-06 to yesterday (the gap between CSV and today)
# ═══════════════════════════════════════════════════════════════════════════════

# Karachi monthly AQI baselines from published environmental reports
MONTHLY_AQI = {
    1: 175, 2: 165, 3: 145, 4: 130, 5: 125,
    6: 110, 7:  90, 8:  85, 9:  95, 10: 120,
    11: 150, 12: 170,
}

# Weekday adjustment — slightly worse traffic/industrial days
DOW_ADJ = {0: +8, 1: +8, 2: +6, 3: +6, 4: +10, 5: +2, 6: 0}


def _synthetic_aqi(date: datetime) -> dict:
    """
    Realistic synthetic AQI for a single date.
    Based on Karachi's known seasonal pollution profile.
    """
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
    """Karachi seasonal weather defaults — used for the synthetic gap."""
    month = date.month
    if month in [12, 1, 2]:
        temp, hum, wind = 18, 55, 3.0
    elif month in [3, 4, 5]:
        temp, hum, wind = 30, 48, 4.0
    elif month in [6, 7, 8, 9]:
        temp, hum, wind = 33, 75, 5.5
    else:
        temp, hum, wind = 26, 58, 3.5

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
    """
    Generates one synthetic row per day between start and end.
    Used to fill the gap between the CSV's last date and today.
    """
    if start > end:
        print("  No synthetic gap to fill — CSV covers up to today.")
        return pd.DataFrame()

    total = (end - start).days + 1
    print(f"  Generating {total} synthetic days "
          f"({start.date()} to {end.date()})...")

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

    feature_df = pd.DataFrame(rows)
    feature_df = compute_features_df(feature_df)
    print(f"  Synthetic gap: {len(feature_df)} rows generated")
    return feature_df


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINE AND UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def run_backfill(csv_path: str, synthetic_start: datetime, synthetic_end: datetime):
    """
    Runs the full 3-layer backfill:
      Layer 1 — Real Kaggle CSV data
      Layer 2 — Synthetic gap from CSV end date to yesterday
      Layer 3 — Handled automatically by the hourly pipeline (not this script)
    """
    all_frames = []

    # ── Layer 1: Kaggle CSV ───────────────────────────────────────────
    if os.path.exists(csv_path):
        print("\n[Layer 1] Loading Kaggle CSV (real data)...")
        csv_df = load_kaggle_csv(csv_path)
        all_frames.append(csv_df)
        print(f"  Layer 1 complete: {len(csv_df)} rows")
    else:
        print(f"\n[Layer 1] CSV not found at {csv_path}")
        print("  Skipping Layer 1 — only synthetic data will be used.")
        print("  To use real data: upload karachi_daily_aqi_weather.csv to data/ folder")

    # ── Layer 2: Synthetic gap ────────────────────────────────────────
    print(f"\n[Layer 2] Generating synthetic gap data...")
    gap_df = generate_synthetic_gap(synthetic_start, synthetic_end)
    if not gap_df.empty:
        all_frames.append(gap_df)
        print(f"  Layer 2 complete: {len(gap_df)} rows")

    # ── Combine ───────────────────────────────────────────────────────
    if not all_frames:
        print("\nNo data generated. Exiting.")
        return

    print("\n[Combining] Merging all layers...")
    combined = pd.concat(all_frames, ignore_index=True)

    # Sort by time and remove any duplicate timestamps
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["timestamp"], keep="first")
    combined.reset_index(drop=True, inplace=True)

    print(f"  Total rows after merge + dedup: {len(combined)}")
    print(f"  Date range: {combined['timestamp'].min().date()} "
          f"to {combined['timestamp'].max().date()}")

    # ── Upload to Hopsworks ───────────────────────────────────────────
    print(f"\n[Uploading] Sending {len(combined)} rows to Hopsworks Feature Store...")
    insert_features(combined)

    print("\n" + "=" * 55)
    print("Backfill complete!")
    print(f"  Layer 1 (Kaggle CSV real data):   up to 2025-08-05")
    print(f"  Layer 2 (Synthetic seasonal data): 2025-08-06 to yesterday")
    print(f"  Layer 3 (Live AQICN hourly data):  from today, runs automatically")
    print("=" * 55)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="3-layer AQI backfill")
    parser.add_argument(
        "--csv",
        default=CSV_PATH,
        help="Path to Kaggle CSV (default: data/karachi_daily_aqi_weather.csv)"
    )
    parser.add_argument(
        "--gap-start",
        default=None,
        help="Start of synthetic gap (default: day after CSV ends = 2025-08-06)"
    )
    parser.add_argument(
        "--gap-end",
        default=None,
        help="End of synthetic gap (default: yesterday)"
    )
    args = parser.parse_args()

    # Synthetic gap runs from the day after CSV ends up to yesterday
    gap_start = (
        datetime.strptime(args.gap_start, "%Y-%m-%d")
        if args.gap_start
        else CSV_END_DATE + timedelta(days=1)
    )
    gap_end = (
        datetime.strptime(args.gap_end, "%Y-%m-%d")
        if args.gap_end
        else datetime.utcnow() - timedelta(days=1)
    )

    print("=" * 55)
    print("AQI Predictor — 3-Layer Historical Backfill")
    print(f"  Layer 1 (Kaggle CSV real data):   up to 2025-08-05")
    print("=" * 55)
    print(f"CSV path:       {args.csv}")
    print(f"Synthetic gap:  {gap_start.date()} to {gap_end.date()}")
    print(f"Live pipeline:  today onwards (automatic)")

    run_backfill(
        csv_path        = args.csv,
        synthetic_start = gap_start,
        synthetic_end   = gap_end,
    )


if __name__ == "__main__":
    main()

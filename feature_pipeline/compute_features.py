"""
feature_pipeline/compute_features.py
======================================
Takes raw readings (from fetch_data.py or a historical CSV row)
and computes the full feature set that the model expects.

Features generated:
  - Time-based: hour, day_of_week, month, is_weekend, season
  - Lag features: AQI_lag_1, AQI_lag_2, AQI_lag_3  (requires history)
  - Change rate: AQI_diff (AQI today - AQI yesterday)
  - Rolling: AQI_rolling_mean_3, AQI_rolling_std_3
  - Pollutant ratios
  - Weather features
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ─── AQI Category helper ────────────────────────────────────────────────────

def aqi_category(aqi: float) -> str:
    """Return WHO AQI category string for a given AQI value."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_season(month: int) -> str:
    """Karachi seasons based on month."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8, 9]:
        return "Summer"
    else:
        return "Autumn"


# ─── Single-row feature computation ─────────────────────────────────────────

def compute_features_single(raw: dict) -> dict:
    """
    Compute features for a single raw reading (no lag history available).
    Lag/rolling features will be NaN — fill them after building a DataFrame.
    Used during backfill row-by-row before calling compute_features_df().
    """
    ts: datetime = raw["timestamp"]

    features = {
        # Time-based
        "timestamp":   ts,
        "hour":        ts.hour,
        "day_of_week": ts.weekday(),          # 0=Monday, 6=Sunday
        "month":       ts.month,
        "day_of_year": ts.timetuple().tm_yday,
        "is_weekend":  int(ts.weekday() >= 5),
        "season":      get_season(ts.month),

        # AQI
        "AQI": raw["AQI"],

        # Pollutants
        "PM2.5": raw["PM2.5"],
        "PM10":  raw["PM10"],
        "NO2":   raw["NO2"],
        "SO2":   raw["SO2"],
        "O3":    raw["O3"],
        "CO":    raw["CO"],

        # Pollutant ratios (useful derived features)
        "PM25_PM10_ratio": raw["PM2.5"] / (raw["PM10"] + 1e-6),
        "NO2_SO2_ratio":   raw["NO2"]   / (raw["SO2"]  + 1e-6),

        # Weather
        "temperature":  raw["temperature"],
        "feels_like":   raw["feels_like"],
        "humidity":     raw["humidity"],
        "pressure":     raw["pressure"],
        "wind_speed":   raw["wind_speed"],
        "wind_deg":     raw["wind_deg"],
        "visibility":   raw["visibility"],
        "cloud_cover":  raw["cloud_cover"],

        # AQI category
        "aqi_category": aqi_category(raw["AQI"]),
    }

    return features


# ─── DataFrame-level feature engineering (adds lags, rolling stats) ──────────

def compute_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of rows (each from compute_features_single),
    sorted by timestamp ascending, add time-series features:
      - Lag features: AQI_lag_1, AQI_lag_2, AQI_lag_3
      - Difference:   AQI_diff
      - Rolling mean & std over 3 periods

    Also one-hot encodes the 'season' column.
    Returns the enriched DataFrame (drops rows where lags are NaN).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lag features
    df["AQI_lag_1"] = df["AQI"].shift(1)
    df["AQI_lag_2"] = df["AQI"].shift(2)
    df["AQI_lag_3"] = df["AQI"].shift(3)

    # AQI change rate (1-step difference)
    df["AQI_diff"] = df["AQI"].diff(1)

    # Rolling statistics (window=3)
    df["AQI_rolling_mean_3"] = df["AQI"].rolling(3).mean()
    df["AQI_rolling_std_3"]  = df["AQI"].rolling(3).std()

    # Target columns for 3-day ahead prediction
    df["AQI_t1"] = df["AQI"].shift(-1)   # next day AQI
    df["AQI_t2"] = df["AQI"].shift(-2)   # day after
    df["AQI_t3"] = df["AQI"].shift(-3)   # 3 days ahead

    # One-hot encode season
    season_dummies = pd.get_dummies(df["season"], prefix="season")
    df = pd.concat([df, season_dummies], axis=1)
    df.drop(columns=["season"], inplace=True)

    # Drop rows with NaN lags (first 3 rows) or NaN targets (last 3 rows)
    df.dropna(subset=["AQI_lag_1", "AQI_lag_2", "AQI_lag_3",
                       "AQI_t1",   "AQI_t2",    "AQI_t3"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  ✅ Feature DataFrame shape after engineering: {df.shape}")
    return df


# ─── Feature column definitions ─────────────────────────────────────────────

FEATURE_COLS = [
    "hour", "day_of_week", "month", "day_of_year", "is_weekend",
    "PM2.5", "PM10", "NO2", "SO2", "O3", "CO",
    "PM25_PM10_ratio", "NO2_SO2_ratio",
    "temperature", "feels_like", "humidity", "pressure",
    "wind_speed", "wind_deg", "visibility", "cloud_cover",
    "AQI_lag_1", "AQI_lag_2", "AQI_lag_3",
    "AQI_diff", "AQI_rolling_mean_3", "AQI_rolling_std_3",
    "season_Autumn", "season_Spring", "season_Summer", "season_Winter",
]

TARGET_COLS = ["AQI_t1", "AQI_t2", "AQI_t3"]


if __name__ == "__main__":
    # Quick test with dummy data
    import json
    from datetime import timedelta

    base = datetime(2024, 1, 1, 12, 0, 0)
    rows = []
    for i in range(10):
        rows.append(compute_features_single({
            "timestamp": base + timedelta(days=i),
            "AQI": 80 + i * 5,
            "PM2.5": 25, "PM10": 50, "NO2": 10, "SO2": 5, "O3": 60, "CO": 300,
            "temperature": 28, "feels_like": 30, "humidity": 70,
            "pressure": 1010, "wind_speed": 3, "wind_deg": 180,
            "visibility": 8, "cloud_cover": 20,
        }))

    df = pd.DataFrame(rows)
    df = compute_features_df(df)
    print(df[["timestamp", "AQI", "AQI_lag_1", "AQI_diff", "AQI_t1"]].to_string())

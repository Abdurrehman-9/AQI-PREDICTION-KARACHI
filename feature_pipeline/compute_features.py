"""
feature_pipeline/compute_features.py
======================================
Takes raw readings (from fetch_data.py or a historical CSV row)
and computes the full feature set that the model expects.

IMPORTANT — Column naming rules:
  Hopsworks requires ALL column names to be:
  - lowercase only
  - no dots (PM2.5 becomes pm25)
  - only letters, numbers, underscores
  - must start with a letter

Features generated:
  - Time-based: hour, day_of_week, month, is_weekend, season
  - Lag features: aqi_lag_1, aqi_lag_2, aqi_lag_3
  - Change rate: aqi_diff (AQI today - AQI yesterday)
  - Rolling: aqi_rolling_mean_3, aqi_rolling_std_3
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
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8, 9]:
        return "summer"
    else:
        return "autumn"


# ─── Single-row feature computation ─────────────────────────────────────────

def compute_features_single(raw: dict) -> dict:
    """
    Compute features for a single raw reading (no lag history available).
    Lag/rolling features will be NaN — fill them after building a DataFrame
    by calling compute_features_df().

    All column names are lowercase with no dots — required by Hopsworks.
    """
    ts: datetime = raw["timestamp"]

    features = {
        # Timestamp
        "timestamp":   ts,

        # Time-based features
        "hour":        ts.hour,
        "day_of_week": ts.weekday(),
        "month":       ts.month,
        "day_of_year": ts.timetuple().tm_yday,
        "is_weekend":  int(ts.weekday() >= 5),
        "season":      get_season(ts.month),

        # AQI (the main target signal — kept uppercase internally,
        # Hopsworks will lowercase it automatically)
        "AQI": raw["AQI"],

        # Pollutants — all lowercase, no dots
        "pm25": raw["PM2.5"],
        "pm10": raw["PM10"],
        "no2":  raw["NO2"],
        "so2":  raw["SO2"],
        "o3":   raw["O3"],
        "co":   raw["CO"],

        # Pollutant ratios
        "pm25_pm10_ratio": raw["PM2.5"] / (raw["PM10"] + 1e-6),
        "no2_so2_ratio":   raw["NO2"]   / (raw["SO2"]  + 1e-6),

        # Weather
        "temperature":  raw["temperature"],
        "feels_like":   raw["feels_like"],
        "humidity":     raw["humidity"],
        "pressure":     raw["pressure"],
        "wind_speed":   raw["wind_speed"],
        "wind_deg":     raw["wind_deg"],
        "visibility":   raw["visibility"],
        "cloud_cover":  raw["cloud_cover"],

        # AQI category (string label for display only)
        "aqi_category": aqi_category(raw["AQI"]),
    }

    return features


# ─── DataFrame-level feature engineering (adds lags, rolling stats) ──────────

def compute_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of rows (each from compute_features_single),
    sorted by timestamp ascending, add time-series features:
      - Lag features: aqi_lag_1, aqi_lag_2, aqi_lag_3
      - Difference:   aqi_diff
      - Rolling mean & std over 3 periods
      - Target columns: aqi_t1, aqi_t2, aqi_t3

    Also one-hot encodes the 'season' column (lowercase values).
    Returns the enriched DataFrame (drops rows where lags are NaN).

    All column names are lowercase — required by Hopsworks.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lag features (all lowercase)
    df["aqi_lag_1"] = df["AQI"].shift(1)
    df["aqi_lag_2"] = df["AQI"].shift(2)
    df["aqi_lag_3"] = df["AQI"].shift(3)

    # AQI change rate (1-step difference)
    df["aqi_diff"] = df["AQI"].diff(1)

    # Rolling statistics (window=3)
    df["aqi_rolling_mean_3"] = df["AQI"].rolling(3).mean()
    df["aqi_rolling_std_3"]  = df["AQI"].rolling(3).std()

    # Target columns for 3-day ahead prediction (all lowercase)
    df["aqi_t1"] = df["AQI"].shift(-1)   # next day AQI
    df["aqi_t2"] = df["AQI"].shift(-2)   # day after
    df["aqi_t3"] = df["AQI"].shift(-3)   # 3 days ahead

    # One-hot encode season (values are already lowercase: winter/spring/summer/autumn)
    season_dummies = pd.get_dummies(df["season"], prefix="season")
    # Ensure all 4 season columns always exist even if season not in data
    for col in ["season_winter", "season_spring", "season_summer", "season_autumn"]:
        if col not in season_dummies.columns:
            season_dummies[col] = 0
    df = pd.concat([df, season_dummies], axis=1)
    df.drop(columns=["season"], inplace=True)

    # Drop rows with NaN lags (first 3 rows) or NaN targets (last 3 rows)
    df.dropna(subset=[
        "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
        "aqi_t1",    "aqi_t2",    "aqi_t3"
    ], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Lowercase the AQI column name to satisfy Hopsworks
    df.rename(columns={"AQI": "aqi"}, inplace=True)

    # Drop string column that Hopsworks can't store easily
    if "aqi_category" in df.columns:
        df.drop(columns=["aqi_category"], inplace=True)

    print(f"  ✅ Feature DataFrame shape after engineering: {df.shape}")
    return df


# ─── Feature column definitions ─────────────────────────────────────────────
# All lowercase — matches Hopsworks storage column names exactly

FEATURE_COLS = [
    "hour", "day_of_week", "month", "day_of_year", "is_weekend",
    "pm25", "pm10", "no2", "so2", "o3", "co",
    "pm25_pm10_ratio", "no2_so2_ratio",
    "temperature", "feels_like", "humidity", "pressure",
    "wind_speed", "wind_deg", "visibility", "cloud_cover",
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    "aqi_diff", "aqi_rolling_mean_3", "aqi_rolling_std_3",
    "season_autumn", "season_spring", "season_summer", "season_winter",
]

TARGET_COLS = ["aqi_t1", "aqi_t2", "aqi_t3"]


if __name__ == "__main__":
    # Quick test with dummy data
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
    print(df[["timestamp", "aqi", "aqi_lag_1", "aqi_diff", "aqi_t1"]].to_string())
    print("\nAll columns:", list(df.columns))

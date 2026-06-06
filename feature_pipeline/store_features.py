"""
feature_pipeline/store_features.py
=====================================
Connects to the Hopsworks Feature Store and:
  - Creates the feature group (if it doesn't exist yet)
  - Inserts new rows into the feature group
  - Reads features back for training

Free tier: https://app.hopsworks.ai
Docs: https://docs.hopsworks.ai/latest/
"""

import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_karachi")

FEATURE_GROUP_NAME    = "aqi_features"
FEATURE_GROUP_VERSION = 1


def get_feature_store():
    """Login to Hopsworks and return the Feature Store object."""
    print("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()
    print(f"  ✅ Connected to project: {HOPSWORKS_PROJECT}")
    return fs


def get_or_create_feature_group(fs):
    """
    Return the feature group, creating it if it doesn't exist.
    The primary key is 'timestamp' — Hopsworks will deduplicate on this.
    """
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp"],
        description="Hourly AQI + weather features for Karachi",
        online_enabled=True,   # enables low-latency reads for serving
    )
    return fg


def insert_features(df: pd.DataFrame) -> None:
    """
    Insert a batch of feature rows into the Hopsworks Feature Store.
    df must contain all columns produced by compute_features.py.
    """
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

    # Hopsworks requires timestamp to be a string or datetime — ensure datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"  ⬆️  Inserting {len(df)} rows into feature group '{FEATURE_GROUP_NAME}'...")
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"  ✅ Insert complete.")


def read_features(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Read all (or a date-range slice of) features from the Feature Store.
    Returns a pandas DataFrame sorted by timestamp ascending.

    Args:
        start_date: ISO date string, e.g. "2023-01-01"
        end_date:   ISO date string, e.g. "2024-12-31"
    """
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

    print(f"  ⬇️  Reading features from '{FEATURE_GROUP_NAME}'...")
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if start_date:
        df = df[df["timestamp"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.Timestamp(end_date)]

    print(f"  ✅ Read {len(df)} rows. Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def read_latest_features(n: int = 7) -> pd.DataFrame:
    """
    Read the most recent n rows — used for real-time inference in the web app.
    """
    df = read_features()
    return df.tail(n).reset_index(drop=True)


if __name__ == "__main__":
    # Quick connectivity test
    df_test = read_latest_features(n=3)
    print(df_test[["timestamp", "AQI", "PM2.5", "temperature"]])

"""
feature_pipeline/store_features.py
"""
import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY     = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT     = os.getenv("HOPSWORKS_PROJECT", "AQI_Pred_Karachi")
HOPSWORKS_HOST        = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
FEATURE_GROUP_NAME    = "aqi_features"
FEATURE_GROUP_VERSION = 2


def get_feature_store():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        host          = HOPSWORKS_HOST,
        project       = HOPSWORKS_PROJECT,
        api_key_value = HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    print(f"  Connected to project: {HOPSWORKS_PROJECT}")
    return fs


def get_or_create_feature_group(fs):
    fg = fs.get_or_create_feature_group(
        name           = FEATURE_GROUP_NAME,
        version        = FEATURE_GROUP_VERSION,
        primary_key    = ["timestamp"],
        description    = "Hourly AQI + weather features for Karachi",
        online_enabled = True,
    )
    return fg


def insert_features(df: pd.DataFrame) -> None:
    df = df.copy()

    # Step 1 — fix timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Step 2 — lowercase all column names
    df.columns = [c.lower() for c in df.columns]

    # Step 3 — convert each column to numeric safely, skip broken ones
    scalar_cols = ["timestamp"]
    for col in df.columns:
        if col == "timestamp":
            continue
        try:
            df[col] = df[col].astype(str).apply(
                lambda x: float(x) if x not in ["nan", "None", "none", ""] else float("nan")
            )
            scalar_cols.append(col)
        except Exception:
            print(f"  Skipping column '{col}' — not convertible")

    df = df[scalar_cols]

    # Step 4 — drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")

    # Step 5 — drop rows missing the main target
    if "aqi" in df.columns:
        df = df.dropna(subset=["aqi"])

    print(f"  Inserting {len(df)} rows | {len(df.columns)} columns")

    fs  = get_feature_store()
    fg  = get_or_create_feature_group(fs)

    fg.insert(
        df,
        write_options={
            "wait_for_job":                  False,
            "start_offline_materialization": True,
        }
    )
    print(f"  Done. Rows written to Hopsworks.")


def read_features(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    fs  = get_feature_store()
    fg  = get_or_create_feature_group(fs)
    print(f"  Reading features from '{FEATURE_GROUP_NAME}'...")

    df = fg.read(online=True)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if start_date:
        df = df[df["timestamp"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.Timestamp(end_date)]

    print(f"  Read {len(df)} rows.")
    return df


def read_latest_features(n: int = 7) -> pd.DataFrame:
    df = read_features()
    return df.tail(n).reset_index(drop=True)

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

# These must stay as integers — Hopsworks schema requires bigint
INT_COLS = [
    "hour", "day_of_week", "month", "day_of_year", "is_weekend",
    "humidity", "wind_deg", "cloud_cover",
]

# These must stay as booleans — Hopsworks schema requires boolean
BOOL_COLS = [
    "season_autumn", "season_spring", "season_summer", "season_winter",
]


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


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and type-corrects a DataFrame before sending to Hopsworks.
    Handles duplicate columns, nested structures, and type mismatches.
    """
    df = df.copy()

    # Step 1 — remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    # Step 2 — lowercase all column names
    df.columns = [c.lower() for c in df.columns]

    # Step 3 — remove duplicates again after lowercasing
    df = df.loc[:, ~df.columns.duplicated()]

    # Step 4 — fix timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Step 5 — flatten any column that is a DataFrame instead of Series
    for col in df.columns:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    # Step 6 — apply correct types per column
    for col in df.columns:
        if col == "timestamp":
            continue
        elif col in INT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        elif col in BOOL_COLS:
            df[col] = df[col].fillna(False).astype(bool)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 7 — ensure all BOOL_COLS exist (add as False if missing)
    for col in BOOL_COLS:
        if col not in df.columns:
            df[col] = False

    # Step 8 — drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")

    # Step 9 — drop rows missing aqi
    if "aqi" in df.columns:
        df = df.dropna(subset=["aqi"])

    return df


def insert_features(df: pd.DataFrame) -> None:
    df = _clean_df(df)

    print(f"  Inserting {len(df)} rows | {len(df.columns)} columns")

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

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
    df = _clean_df(df)
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

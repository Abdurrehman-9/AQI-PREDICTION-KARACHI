"""
feature_pipeline/store_features.py
=====================================
Connects to the Hopsworks Feature Store and utilizes the instant 
Online Store to completely bypass background server queues.
"""

import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "AQI_Pred_Karachi")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

FEATURE_GROUP_NAME    = "aqi_features"
FEATURE_GROUP_VERSION = 2


def get_feature_store():
    """Login to Hopsworks and return the Feature Store object."""
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
        name        = FEATURE_GROUP_NAME,
        version     = FEATURE_GROUP_VERSION,
        primary_key = ["timestamp"],
        description = "Hourly AQI + weather features for Karachi",
        online_enabled = True, 
    )
    return fg


def insert_features(df: pd.DataFrame) -> None:
    # 1. Force the timestamp to standard datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 2. Standardize column naming to lowercase
    df.columns = df.columns.str.lower()
    
    # 3. Robust numeric conversion: 
    # Select all columns except timestamp, apply to_numeric to the whole subset at once
    cols_to_convert = df.columns.drop("timestamp")
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    
    # 4. Clean up any columns that became entirely NaN
    df = df.dropna(axis=1, how='all')

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    
    print(f"  Inserting {len(df)} rows into '{FEATURE_GROUP_NAME}'...")
    
    # 5. Perform the insertion
    fg.insert(
        df,
        write_options={
            "wait_for_job": False,
            "start_offline_materialization": True 
        }
    )
    
    print(f"🚀 Injection complete! Rows safely written to the Online Store.")


def read_features(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    print(f"  Reading features from '{FEATURE_GROUP_NAME}' Online DB...")
    
    df = fg.read(online=True)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.columns = df.columns.str.lower()
    
    if start_date:
        df = df[df["timestamp"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.Timestamp(end_date)]
    print(f"  Read {len(df)} rows from Online Store.")
    return df


def read_latest_features(n: int = 7) -> pd.DataFrame:
    df = read_features()
    return df.tail(n).reset_index(drop=True)

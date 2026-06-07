"""
feature_pipeline/store_features.py
=====================================
Connects to the Hopsworks Feature Store and writes data safely.
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
FEATURE_GROUP_VERSION = 1


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
        online_enabled = False,
    )
    return fg


def insert_features(df: pd.DataFrame) -> None:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Clean column names before sending to Hopsworks storage
    rename_dict = {col: col.replace("PM2.5", "pm2_5").replace("pm2.5", "pm2_5") for col in df.columns if "2.5" in col}
    if rename_dict:
        df = df.rename(columns=rename_dict)
        
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    
    print(f"  Inserting {len(df)} rows into '{FEATURE_GROUP_NAME}'...")
    
    # 🚀 THE OPTIMIZED SETTING:
    # wait_for_job=False stops GitHub from hanging and timing out.
    # start_offline_materialization=True tells the cluster to process it.
    fg.insert(
        df,
        write_options={
            "wait_for_job": False,
            "start_offline_materialization": True
        }
    )
    
    print(f"🚀 Data uploaded successfully! Hopsworks is materializing rows in the background.")


def read_features(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    print(f"  Reading features from '{FEATURE_GROUP_NAME}'...")
    df = fg.read()
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

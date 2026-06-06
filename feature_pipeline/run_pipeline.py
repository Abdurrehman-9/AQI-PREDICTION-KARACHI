"""
feature_pipeline/run_pipeline.py
===================================
Orchestrates the full feature pipeline:
  1. Fetch raw data from APIs
  2. Compute features
  3. Store in Hopsworks Feature Store

This script is run:
  - Every hour via GitHub Actions
  - Every hour via Airflow DAG
"""

import pandas as pd
from fetch_data import get_raw_reading
from compute_features import compute_features_single, compute_features_df
from store_features import read_latest_features, insert_features


def run():
    print("=" * 55)
    print("🌫️  AQI Feature Pipeline — Starting")
    print("=" * 55)

    # ── Step 1: Fetch raw data ──────────────────────────────
    print("\n[1/3] Fetching raw data from APIs...")
    raw = get_raw_reading()

    # ── Step 2: Compute features for the new row ────────────
    print("\n[2/3] Computing features...")
    new_row = compute_features_single(raw)

    # To compute lag/rolling features, we need to prepend
    # recent history from the Feature Store
    print("      Loading recent history for lag computation...")
    try:
        history_df = read_latest_features(n=10)
        # Drop target columns if present (they won't be in live data)
        for col in ["AQI_t1", "AQI_t2", "AQI_t3"]:
            if col in history_df.columns:
                history_df.drop(columns=[col], inplace=True)
    except Exception as e:
        print(f"      ⚠️  Could not load history (first run?): {e}")
        history_df = pd.DataFrame()

    # Combine history + new row, then recompute time-series features
    new_df  = pd.DataFrame([new_row])
    combined = pd.concat([history_df, new_df], ignore_index=True)
    enriched = compute_features_df(combined)

    # We only want to store the NEW row (the last one after enrichment)
    latest = enriched.tail(1).reset_index(drop=True)
    print(f"      New feature row: {latest[['timestamp', 'AQI', 'AQI_lag_1', 'AQI_diff']].to_dict('records')}")

    # ── Step 3: Store in Feature Store ─────────────────────
    print("\n[3/3] Storing features in Hopsworks Feature Store...")
    insert_features(latest)

    print("\n✅ Feature pipeline complete!\n")


if __name__ == "__main__":
    run()

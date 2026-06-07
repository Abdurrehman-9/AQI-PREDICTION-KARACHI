"""
feature_pipeline/run_pipeline.py
===================================
Orchestrates the full feature pipeline:
  1. Fetch raw data from APIs
  2. Compute features
  3. Store in Hopsworks Feature Store

Run every hour via GitHub Actions.
"""

import pandas as pd
from fetch_data import get_raw_reading
from compute_features import compute_features_single, compute_features_df, FEATURE_COLS, TARGET_COLS
from store_features import read_latest_features, insert_features


def run():
    print("=" * 55)
    print("🌫️  AQI Feature Pipeline — Starting")
    print("=" * 55)

    # ── Step 1: Fetch raw data ──────────────────────────────
    print("\n[1/3] Fetching raw data from APIs...")
    raw = get_raw_reading()

    # ── Step 2: Compute features ────────────────────────────
    print("\n[2/3] Computing features...")
    new_row = compute_features_single(raw)

    print("      Loading recent history for lag computation...")
    try:
        history_df = read_latest_features(n=10)
        # Keep only the columns we need — avoids duplicate column issues
        keep = ["timestamp", "aqi"] + FEATURE_COLS
        keep = [c for c in keep if c in history_df.columns]
        history_df = history_df[keep].copy()
    except Exception as e:
        print(f"      Could not load history (first run?): {e}")
        history_df = pd.DataFrame()

    # Build new row as DataFrame — lowercase everything immediately
    new_df = pd.DataFrame([new_row])
    new_df.columns = [c.lower() for c in new_df.columns]

    # Rename AQI to aqi if present (from compute_features_single)
    if "aqi" not in new_df.columns and "AQI" in new_df.columns:
        new_df.rename(columns={"AQI": "aqi"}, inplace=True)

    # Drop columns that only exist in history (targets etc)
    for col in TARGET_COLS + ["aqi_t1", "aqi_t2", "aqi_t3"]:
        if col in new_df.columns:
            new_df.drop(columns=[col], inplace=True)
        if not history_df.empty and col in history_df.columns:
            history_df.drop(columns=[col], inplace=True)

    # Align columns between history and new row before concat
    if not history_df.empty:
        history_df.columns = [c.lower() for c in history_df.columns]
        # Only keep columns that exist in both
        common_cols = [c for c in history_df.columns if c in new_df.columns]
        history_df  = history_df[common_cols]
        new_df      = new_df[common_cols]

    # Combine and compute lag features
    combined = pd.concat([history_df, new_df], ignore_index=True)

    # Remove any duplicate columns before feature engineering
    combined = combined.loc[:, ~combined.columns.duplicated()]

    enriched = compute_features_df(combined)

    # Remove duplicate columns again after engineering (safety)
    enriched = enriched.loc[:, ~enriched.columns.duplicated()]

    # Take only the last row — the new one
    latest = enriched.tail(1).reset_index(drop=True)

    print(f"      New feature row: "
          f"{latest[['timestamp', 'aqi', 'aqi_lag_1', 'aqi_diff']].to_dict('records')}")

    # ── Step 3: Store in Feature Store ─────────────────────
    print("\n[3/3] Storing features in Hopsworks Feature Store...")
    insert_features(latest)

    print("\n✅ Feature pipeline complete!\n")


if __name__ == "__main__":
    run()

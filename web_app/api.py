"""
web_app/api.py
================
FastAPI backend — exposes JSON endpoints for the AQI dashboard
and for external consumers.

Endpoints:
  GET /current      → latest AQI reading
  GET /forecast     → 3-day AQI forecast
  GET /history      → historical AQI data (last N days)
  GET /health       → health check

Run:
    uvicorn web_app.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

app = FastAPI(
    title="Karachi AQI Predictor API",
    description="Real-time and forecast AQI data for Karachi, Pakistan",
    version="1.0.0",
)

# Allow Streamlit to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALERT_THRESHOLD = int(os.getenv("ALERT_AQI_THRESHOLD", 150))


# ─── AQI category helper ─────────────────────────────────────────────────────

def aqi_category(aqi: float) -> dict:
    levels = [
        (0,   50,  "Good",                         "#00e400"),
        (51,  100, "Moderate",                     "#ffff00"),
        (101, 150, "Unhealthy for Sensitive",       "#ff7e00"),
        (151, 200, "Unhealthy",                     "#ff0000"),
        (201, 300, "Very Unhealthy",                "#8f3f97"),
        (301, 500, "Hazardous",                     "#7e0023"),
    ]
    for lo, hi, label, color in levels:
        if lo <= aqi <= hi:
            return {"label": label, "color": color, "alert": aqi >= ALERT_THRESHOLD}
    return {"label": "Hazardous", "color": "#7e0023", "alert": True}


# ─── Response models ──────────────────────────────────────────────────────────

class AQIReading(BaseModel):
    timestamp:   str
    aqi:         float
    category:    str
    color:       str
    alert:       bool
    pm25:        float
    pm10:        float
    no2:         float
    so2:         float
    o3:          float
    co:          float
    temperature: float
    humidity:    float
    wind_speed:  float


class ForecastDay(BaseModel):
    date:     str
    day:      int
    aqi:      float
    category: str
    color:    str
    alert:    bool


class ForecastResponse(BaseModel):
    generated_at: str
    forecast:     list[ForecastDay]


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/current", response_model=AQIReading)
def get_current():
    """Fetch the latest real-time AQI reading."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))
        from fetch_data import get_raw_reading
        raw = get_raw_reading()
        cat = aqi_category(raw["AQI"])
        return AQIReading(
            timestamp   = raw["timestamp"].isoformat(),
            aqi         = round(raw["AQI"], 1),
            category    = cat["label"],
            color       = cat["color"],
            alert       = cat["alert"],
            pm25        = round(raw.get("PM2.5", 0), 1),
            pm10        = round(raw.get("PM10",  0), 1),
            no2         = round(raw.get("NO2",   0), 1),
            so2         = round(raw.get("SO2",   0), 1),
            o3          = round(raw.get("O3",    0), 1),
            co          = round(raw.get("CO",    0), 1),
            temperature = round(raw.get("temperature", 0), 1),
            humidity    = round(raw.get("humidity",    0), 1),
            wind_speed  = round(raw.get("wind_speed",  0), 1),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast():
    """Return the 3-day AQI forecast from the trained model."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))
        from store_features import read_latest_features
        from compute_features import FEATURE_COLS

        history = read_latest_features(n=30)
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        report_path = os.path.join(models_dir, "training_report.json")

        if not os.path.exists(report_path):
            raise HTTPException(status_code=503, detail="Model not trained yet. Run training pipeline first.")

        with open(report_path) as f:
            report = json.load(f)

        best_name = report["best_model"]
        model_dir = os.path.join(models_dir, best_name.lower().replace(" ", "_"))

        import joblib
        model  = joblib.load(os.path.join(model_dir, "model.pkl"))
        X      = history[FEATURE_COLS].tail(1).values
        preds  = model.predict(X)[0]

        forecast = []
        for i, pred in enumerate(preds[:3]):
            date = (datetime.utcnow() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            cat  = aqi_category(pred)
            forecast.append(ForecastDay(
                date     = date,
                day      = i + 1,
                aqi      = round(float(pred), 1),
                category = cat["label"],
                color    = cat["color"],
                alert    = cat["alert"],
            ))

        return ForecastResponse(
            generated_at = datetime.utcnow().isoformat(),
            forecast     = forecast,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def get_history(days: int = Query(default=30, ge=1, le=365)):
    """Return the last N days of historical AQI data."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))
        from store_features import read_features
        df = read_features()
        df = df.tail(days)[["timestamp", "AQI", "PM2.5", "PM10",
                              "temperature", "humidity"]].copy()
        df["timestamp"] = df["timestamp"].astype(str)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

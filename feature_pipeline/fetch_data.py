"""
feature_pipeline/fetch_data.py
================================
Fetches raw AQI and weather data for Karachi from:
  - AQICN API  → pollutants (PM2.5, PM10, NO2, SO2, O3, CO) + AQI
  - OpenWeatherMap API → temperature, humidity, wind speed, pressure

Returns a single dict of raw readings for the current hour.
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN       = os.getenv("AQICN_TOKEN")
OWM_API_KEY       = os.getenv("OWM_API_KEY")
KARACHI_STATION   = os.getenv("KARACHI_STATION_ID", "@karachi")
KARACHI_LAT       = float(os.getenv("KARACHI_LAT", "24.8607"))
KARACHI_LON       = float(os.getenv("KARACHI_LON", "67.0011"))


def fetch_aqicn_data(station: str = KARACHI_STATION) -> dict:
    """
    Call the AQICN feed API.
    Returns raw JSON with AQI + individual pollutant iaqi values.
    Docs: https://aqicn.org/json-api/doc/
    """
    url = f"https://api.waqi.info/feed/{station}/"
    params = {"token": AQICN_TOKEN}

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()
    if data.get("status") != "ok":
        raise ValueError(f"AQICN API error: {data.get('data', 'unknown error')}")

    return data["data"]


def fetch_weather_data(lat: float = KARACHI_LAT, lon: float = KARACHI_LON) -> dict:
    """
    Call OpenWeatherMap current weather API.
    Returns temperature, humidity, wind speed, pressure, description.
    Docs: https://openweathermap.org/current
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OWM_API_KEY,
        "units": "metric",   # Celsius
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    return response.json()


def get_raw_reading() -> dict:
    """
    Combines AQICN + OpenWeatherMap into one flat dict of raw values.
    This is what gets passed to compute_features.py.
    """
    timestamp = datetime.utcnow()
    print(f"[{timestamp.isoformat()}] Fetching AQICN data...")
    aqicn  = fetch_aqicn_data()

    print(f"[{timestamp.isoformat()}] Fetching weather data...")
    weather = fetch_weather_data()

    # --- Parse AQICN pollutants ---
    iaqi = aqicn.get("iaqi", {})

    def safe_get(d: dict, key: str, default: float = 0.0) -> float:
        """Safely extract a value from AQICN iaqi dict."""
        return float(d.get(key, {}).get("v", default))

    raw = {
        # Timestamp
        "timestamp": timestamp,

        # AQI (composite index)
        "AQI": float(aqicn.get("aqi", 0)),

        # Individual pollutants (µg/m³ or ppb depending on AQICN station)
        "PM2.5": safe_get(iaqi, "pm25"),
        "PM10":  safe_get(iaqi, "pm10"),
        "NO2":   safe_get(iaqi, "no2"),
        "SO2":   safe_get(iaqi, "so2"),
        "O3":    safe_get(iaqi, "o3"),
        "CO":    safe_get(iaqi, "co"),

        # Weather from OpenWeatherMap
        "temperature":  weather["main"]["temp"],
        "feels_like":   weather["main"]["feels_like"],
        "humidity":     weather["main"]["humidity"],
        "pressure":     weather["main"]["pressure"],
        "wind_speed":   weather["wind"]["speed"],
        "wind_deg":     weather["wind"].get("deg", 0),
        "visibility":   weather.get("visibility", 10000) / 1000,   # convert m → km
        "cloud_cover":  weather["clouds"]["all"],                  # percent
        "weather_desc": weather["weather"][0]["description"],
    }

    print(f"  ✅ AQI = {raw['AQI']}  |  PM2.5 = {raw['PM2.5']}  |  Temp = {raw['temperature']}°C")
    return raw


if __name__ == "__main__":
    reading = get_raw_reading()
    print("\n📦 Raw reading:")
    for k, v in reading.items():
        print(f"  {k}: {v}")

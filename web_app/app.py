"""
web_app/app.py
================
Karachi AQI Predictor — Streamlit Dashboard
Design: Dark theme, orange AQI banner, pollutant breakdown,
        3-day forecast cards, historical charts, model insights.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st

warnings.filterwarnings("ignore")

# Allow imports from feature_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── AQI Levels ──────────────────────────────────────────────────────────────
AQI_LEVELS = [
    (0,   50,  "#2ecc71", "#1a5c36", "Good",                           "Air quality is satisfactory. No health risk.", "😊"),
    (51,  100, "#f1c40f", "#7d6608", "Moderate",                       "Acceptable air quality. Minor concern for sensitive people.", "😐"),
    (101, 150, "#e67e22", "#784212", "Unhealthy for Sensitive Groups",  "Sensitive groups may experience health effects.", "😤"),
    (151, 200, "#e74c3c", "#7b241c", "Unhealthy",                      "Everyone may begin to experience health effects.", "😷"),
    (201, 300, "#8e44ad", "#4a235a", "Very Unhealthy",                  "Health alert: serious effects for everyone.", "🤢"),
    (301, 500, "#7e0023", "#3d0012", "Hazardous",                      "Emergency conditions. Entire population affected.", "☠️"),
]

def aqi_info(aqi: float) -> tuple:
    for lo, hi, color, dark, label, desc, emoji in AQI_LEVELS:
        if lo <= aqi <= hi:
            return color, dark, label, desc, emoji
    return "#7e0023", "#3d0012", "Hazardous", "Emergency conditions.", "☠️"

ALERT_THRESHOLD = int(os.getenv("ALERT_AQI_THRESHOLD", 150))

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global dark background */
    .stApp { background-color: #0d0d0d; color: #ffffff; }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Top description bar */
    .top-bar {
        background: #1a1a1a;
        padding: 12px 24px;
        text-align: center;
        color: #aaa;
        font-size: 14px;
        margin-bottom: 20px;
        border-bottom: 1px solid #333;
    }
    
    /* AQI Banner */
    .aqi-banner {
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .aqi-banner-left h2 {
        margin: 0 0 6px 0;
        font-size: 26px;
        font-weight: 800;
    }
    .aqi-banner-left p {
        margin: 0;
        font-size: 15px;
        opacity: 0.9;
    }
    .aqi-banner-right {
        text-align: right;
        font-size: 14px;
    }
    .aqi-emoji {
        font-size: 64px;
        margin-bottom: 4px;
    }

    /* Forecast cards */
    .forecast-card {
        border-radius: 12px;
        padding: 20px 24px;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .forecast-card h3 { margin: 0 0 4px 0; font-size: 18px; font-weight: 700; }
    .forecast-card .aqi-val { font-size: 22px; font-weight: 800; margin: 4px 0; }
    .forecast-card .cat { font-size: 13px; font-weight: 600; opacity: 0.9; }
    .forecast-card .pollutants { font-size: 12px; opacity: 0.8; margin-top: 4px; }
    .forecast-card .card-emoji { font-size: 42px; }

    /* Section titles */
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin: 28px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #333;
    }

    /* Weather metrics */
    .weather-strip {
        background: #1a1a1a;
        border-radius: 10px;
        padding: 16px 24px;
        display: flex;
        gap: 40px;
        margin-bottom: 24px;
        border: 1px solid #333;
    }
    .weather-item { text-align: center; }
    .weather-item .val { font-size: 22px; font-weight: 700; color: #fff; }
    .weather-item .lbl { font-size: 12px; color: #888; margin-top: 2px; }

    /* Alert banner */
    .alert-banner {
        background: linear-gradient(90deg, #c0392b, #e74c3c);
        border-radius: 10px;
        padding: 14px 24px;
        text-align: center;
        font-size: 17px;
        font-weight: 700;
        color: white;
        margin-bottom: 20px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.75} }

    /* Sidebar */
    .css-1d391kg { background: #111 !important; }

    /* Metric overrides */
    [data-testid="metric-container"] {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Top description bar ─────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    Real-time air quality monitoring App. Fetches live AQI data every hour and delivers
    accurate 3-day forecasts using the latest trained time-series forecasting model from Hopsworks.
</div>
""", unsafe_allow_html=True)

# ─── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌫️ Karachi AQI")
    st.markdown("**Navigation**")
    page = st.radio("", [
        "🏠 Dashboard",
        "📊 EDA & Trends",
        "🧠 Model Insights",
        "ℹ️ About",
    ], label_visibility="collapsed")
    st.divider()
    st.caption(f"Updated: {datetime.utcnow().strftime('%H:%M UTC')}")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ─── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching live AQI...")
def fetch_live():
    try:
        from fetch_data import get_raw_reading
        return get_raw_reading()
    except Exception as e:
        st.warning(f"Live API unavailable ({e}) — showing demo data.")
        return {
            "timestamp": datetime.utcnow(),
            "AQI": 149.0, "PM2.5": 54.91, "PM10": 98.45,
            "NO2": 18.3,  "SO2": 6.2,     "O3": 52.1, "CO": 320.0,
            "temperature": 31.5, "feels_like": 35.0,
            "humidity": 62,      "pressure": 1008,
            "wind_speed": 3.2,   "wind_deg": 195,
            "visibility": 6.5,   "cloud_cover": 20,
        }


@st.cache_data(ttl=3600, show_spinner="Loading historical data...")
def fetch_history():
    try:
        from store_features import read_features
        df = read_features()
        # Normalise column names
        df.columns = [c.lower() for c in df.columns]
        return df.tail(365)
    except Exception as e:
        st.warning(f"Could not load history ({e}) — showing demo data.")
        dates = pd.date_range(end=datetime.utcnow(), periods=365, freq="D")
        np.random.seed(42)
        aqi = 100 + 60 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 12, 365)
        return pd.DataFrame({
            "timestamp": dates,
            "aqi":   np.clip(aqi, 30, 280),
            "pm25":  np.clip(aqi * 0.38, 0, None),
            "pm10":  np.clip(aqi * 0.65, 0, None),
            "no2":   np.clip(aqi * 0.14, 0, None),
            "so2":   np.clip(aqi * 0.05, 0, None),
            "o3":    np.clip(aqi * 0.30, 0, None),
            "co":    np.clip(aqi * 2.8,  0, None),
            "temperature": 28 + np.random.normal(0, 5, 365),
            "humidity":    65 + np.random.normal(0, 8, 365),
        })


@st.cache_resource(show_spinner="Loading model...")
def fetch_model():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    report_path = os.path.join(models_dir, "training_report.json")
    if not os.path.exists(report_path):
        return None, None, None
    with open(report_path) as f:
        report = json.load(f)
    try:
        import joblib
        best = report["best_model"]
        mdir = os.path.join(models_dir, best.lower().replace(" ", "_"))
        model  = joblib.load(os.path.join(mdir, "model.pkl"))
        scaler = None
        sp = os.path.join(mdir, "scaler.pkl")
        if os.path.exists(sp):
            scaler = joblib.load(sp)
        return model, scaler, report
    except Exception:
        return None, None, report


def get_forecast(model, scaler, history_df):
    """Predict next 3 days AQI."""
    try:
        from compute_features import FEATURE_COLS
        avail = [c for c in FEATURE_COLS if c in history_df.columns]
        X = history_df[avail].tail(1).values
        if scaler:
            X = scaler.transform(X)
        preds = model.predict(X)[0]
        return [max(0, float(p)) for p in preds[:3]]
    except Exception:
        last = float(history_df["aqi"].iloc[-1]) if "aqi" in history_df.columns else 120
        return [last * 0.98, last * 1.02, last * 0.95]


# ─── Load data ────────────────────────────────────────────────────────────────
raw        = fetch_live()
history_df = fetch_history()
model, scaler, report = fetch_model()

current_aqi = float(raw.get("AQI", 0))
color, dark_color, label, desc, emoji = aqi_info(current_aqi)
pm25 = float(raw.get("PM2.5", 0))
pm10 = float(raw.get("PM10",  0))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":

    # ── Hazard alert (only when above threshold) ──────────────────────
    if current_aqi >= ALERT_THRESHOLD:
        st.markdown(f"""
        <div class="alert-banner">
            🚨 HAZARD ALERT — AQI {current_aqi:.0f} in Karachi is {label}.
            Avoid outdoor activity. Wear a mask if going outside.
        </div>
        """, unsafe_allow_html=True)

    # ── Main AQI Banner ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="aqi-banner" style="background: linear-gradient(135deg, {color}, {dark_color});">
        <div class="aqi-banner-left">
            <h2>Today's AQI: {current_aqi:.0f} — {label}</h2>
            <p>{desc}</p>
        </div>
        <div class="aqi-banner-right">
            <div class="aqi-emoji">{emoji}</div>
            <div><strong>Main Pollutant: PM2.5</strong></div>
            <div>PM2.5: {pm25:.2f} µg/m³ | PM10: {pm10:.2f} µg/m³</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Weather strip ─────────────────────────────────────────────────
    wc = st.columns(6)
    weather_items = [
        ("🌡️", f"{raw.get('temperature', 0):.1f}°C", "Temperature"),
        ("💧", f"{raw.get('humidity', 0):.0f}%",      "Humidity"),
        ("💨", f"{raw.get('wind_speed', 0):.1f} m/s", "Wind Speed"),
        ("👁️", f"{raw.get('visibility', 0):.1f} km",  "Visibility"),
        ("🌀", f"{raw.get('pressure', 0):.0f} hPa",   "Pressure"),
        ("☁️", f"{raw.get('cloud_cover', 0):.0f}%",   "Cloud Cover"),
    ]
    for col, (icon, val, lbl) in zip(wc, weather_items):
        col.metric(f"{icon} {lbl}", val)

    # ── Pollutant breakdown bar chart ─────────────────────────────────
    st.markdown('<div class="section-title">Today\'s Pollutant Breakdown</div>', unsafe_allow_html=True)

    pollutants = {
        "PM2.5": float(raw.get("PM2.5", 0)),
        "PM10":  float(raw.get("PM10",  0)),
        "NO2":   float(raw.get("NO2",   0)),
        "SO2":   float(raw.get("SO2",   0)),
        "CO":    float(raw.get("CO",    0)),
        "O3":    float(raw.get("O3",    0)),
    }

    poll_colors = {
        "PM2.5": "#9b59b6",
        "PM10":  "#1abc9c",
        "NO2":   "#3498db",
        "SO2":   "#f39c12",
        "CO":    "#e91e8c",
        "O3":    "#2ecc71",
    }

    fig_bar = go.Figure()
    for pollutant, value in pollutants.items():
        fig_bar.add_trace(go.Bar(
            y=[pollutant], x=[value],
            orientation="h",
            name=pollutant,
            marker_color=poll_colors[pollutant],
            text=f"{value:.2f}",
            textposition="outside",
        ))

    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        height=320,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5),
        xaxis_title="Value",
        yaxis_title="Pollutant",
        margin=dict(l=80, r=120, t=20, b=40),
        barmode="overlay",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── 3-Day Forecast Cards ──────────────────────────────────────────
    st.markdown('<div class="section-title">Next 3 Days AQI Forecast</div>', unsafe_allow_html=True)

    forecast = get_forecast(model, scaler, history_df) if model else \
               [current_aqi * 0.98, current_aqi * 1.02, current_aqi * 0.96]

    fc_cols = st.columns(3)
    for i, (col, pred) in enumerate(zip(fc_cols, forecast)):
        fc_color, fc_dark, fc_label, _, fc_emoji = aqi_info(pred)
        future_date = (datetime.utcnow() + timedelta(days=i+1)).strftime("%A, %d %b")
        with col:
            st.markdown(f"""
            <div class="forecast-card" style="background: linear-gradient(135deg, {fc_color}, {fc_dark});">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <h3>Day {i+1}</h3>
                        <div style="font-size:12px;opacity:0.8;margin-bottom:8px">{future_date}</div>
                        <div class="aqi-val">AQI: {pred:.0f}</div>
                        <div class="cat">{fc_label}</div>
                        <div class="pollutants">PM2.5: {pm25:.2f} µg/m³ | PM10: {pm10:.2f} µg/m³</div>
                    </div>
                    <div class="card-emoji">{fc_emoji}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Historical AQI trend ──────────────────────────────────────────
    st.markdown('<div class="section-title">Karachi Historical Air Quality Data</div>', unsafe_allow_html=True)

    aqi_col = "aqi" if "aqi" in history_df.columns else "AQI"
    ts_col  = "timestamp"
    recent  = history_df.tail(90).copy()

    fig_trend = go.Figure()

    # WHO threshold bands
    for lo, hi, clr, _, lbl, _, _ in AQI_LEVELS:
        fig_trend.add_hrect(
            y0=lo, y1=min(hi, 320),
            fillcolor=clr, opacity=0.07,
            annotation_text=lbl,
            annotation_position="right",
            annotation_font_size=9,
            annotation_font_color="#aaa",
        )

    fig_trend.add_trace(go.Scatter(
        x=recent[ts_col],
        y=recent[aqi_col],
        name="AQI",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}",
    ))

    # Forecast extension
    forecast_dates = [datetime.utcnow() + timedelta(days=i+1) for i in range(3)]
    fig_trend.add_trace(go.Scatter(
        x=forecast_dates, y=forecast,
        name="3-Day Forecast",
        mode="lines+markers",
        line=dict(color="#ffffff", width=2, dash="dash"),
        marker=dict(size=10, symbol="diamond", color="#ffffff"),
    ))

    fig_trend.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#111111",
        height=380,
        xaxis_title="Date",
        yaxis_title="AQI",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=60, t=20, b=60),
    )
    st.plotly_chart(fig_trend, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & TRENDS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 EDA & Trends":
    st.markdown("## 📊 Exploratory Data Analysis")

    df = history_df.copy()
    df.columns = [c.lower() for c in df.columns]

    aqi_col = "aqi" if "aqi" in df.columns else None
    if aqi_col is None:
        st.error("No AQI data available.")
        st.stop()

    df["month"]       = pd.to_datetime(df["timestamp"]).dt.month
    df["dow"]         = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df["month_name"]  = pd.to_datetime(df["timestamp"]).dt.strftime("%b")
    df["year"]        = pd.to_datetime(df["timestamp"]).dt.year

    # ── Full AQI timeline ──────────────────────────────────────────────
    st.markdown('<div class="section-title">AQI Timeline with WHO Bands</div>', unsafe_allow_html=True)
    fig1 = go.Figure()
    for lo, hi, clr, _, lbl, _, _ in AQI_LEVELS:
        fig1.add_hrect(y0=lo, y1=min(hi, 320), fillcolor=clr, opacity=0.07,
                       annotation_text=lbl, annotation_position="right",
                       annotation_font_size=9)
    fig1.add_trace(go.Scatter(
        x=df["timestamp"], y=df[aqi_col],
        line=dict(color="#e67e22", width=1.5), name="AQI",
    ))
    fig1.update_layout(template="plotly_dark", paper_bgcolor="#0d0d0d",
                       plot_bgcolor="#111", height=380,
                       xaxis_title="Date", yaxis_title="AQI")
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)

    # ── Monthly avg ────────────────────────────────────────────────────
    with c1:
        st.markdown('<div class="section-title">Monthly Average AQI</div>', unsafe_allow_html=True)
        mon = df.groupby("month_name")[aqi_col].mean().reset_index()
        order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        mon["month_name"] = pd.Categorical(mon["month_name"], categories=order, ordered=True)
        mon = mon.sort_values("month_name")
        fig2 = px.bar(mon, x="month_name", y=aqi_col,
                      color=aqi_col, color_continuous_scale="RdYlGn_r",
                      template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                           height=300, showlegend=False,
                           xaxis_title="Month", yaxis_title="Avg AQI")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Day of week avg ────────────────────────────────────────────────
    with c2:
        st.markdown('<div class="section-title">AQI by Day of Week</div>', unsafe_allow_html=True)
        dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        dow = df.groupby("dow")[aqi_col].mean().reset_index()
        dow["day"] = dow["dow"].map(lambda x: dow_names[x])
        fig3 = px.bar(dow, x="day", y=aqi_col,
                      color=aqi_col, color_continuous_scale="RdYlGn_r",
                      template="plotly_dark")
        fig3.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                           height=300, showlegend=False,
                           xaxis_title="Day", yaxis_title="Avg AQI")
        st.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns(2)

    # ── Pollutant pie ──────────────────────────────────────────────────
    with c3:
        st.markdown('<div class="section-title">Pollutant Contribution</div>', unsafe_allow_html=True)
        poll_map = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2",
                    "so2": "SO2", "o3": "O3", "co": "CO"}
        avgs = {v: float(df[k].mean()) for k, v in poll_map.items() if k in df.columns}
        fig4 = go.Figure(go.Pie(
            labels=list(avgs.keys()),
            values=list(avgs.values()),
            hole=0.45,
            marker_colors=["#9b59b6","#1abc9c","#3498db","#f39c12","#e91e8c","#2ecc71"],
        ))
        fig4.update_layout(template="plotly_dark", paper_bgcolor="#0d0d0d",
                           height=300, showlegend=True)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Correlation heatmap ────────────────────────────────────────────
    with c4:
        st.markdown('<div class="section-title">Feature Correlation</div>', unsafe_allow_html=True)
        corr_cols = [aqi_col] + [k for k in ["pm25","pm10","no2","so2","o3","co",
                                               "temperature","humidity"] if k in df.columns]
        corr = df[corr_cols].corr()
        fig5 = px.imshow(corr, text_auto=".2f",
                         color_continuous_scale="RdBu_r",
                         template="plotly_dark", aspect="auto")
        fig5.update_layout(paper_bgcolor="#0d0d0d", height=300)
        st.plotly_chart(fig5, use_container_width=True)

    # ── AQI distribution ──────────────────────────────────────────────
    st.markdown('<div class="section-title">AQI Distribution</div>', unsafe_allow_html=True)
    fig6 = px.histogram(df, x=aqi_col, nbins=50,
                        color_discrete_sequence=["#e67e22"],
                        template="plotly_dark")
    fig6.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                       height=300, xaxis_title="AQI", yaxis_title="Count")
    for lo, hi, clr, _, lbl, _, _ in AQI_LEVELS:
        fig6.add_vrect(x0=lo, x1=min(hi, 320), fillcolor=clr, opacity=0.07,
                       annotation_text=lbl, annotation_position="top left",
                       annotation_font_size=8)
    st.plotly_chart(fig6, use_container_width=True)

    # ── Scatter: AQI vs Temperature ────────────────────────────────────
    if "temperature" in df.columns:
        st.markdown('<div class="section-title">AQI vs Temperature</div>', unsafe_allow_html=True)
        fig7 = px.scatter(df, x="temperature", y=aqi_col,
                          color=aqi_col, color_continuous_scale="RdYlGn_r",
                          opacity=0.6, trendline="ols",
                          template="plotly_dark")
        fig7.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                           height=350, xaxis_title="Temperature (°C)",
                           yaxis_title="AQI")
        st.plotly_chart(fig7, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 Model Insights":
    st.markdown("## 🧠 Model Performance & Explainability")

    if report:
        best_name = report.get("best_model", "Unknown")
        avg_rmse  = report.get("avg_rmse", 0)

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e67e22, #784212);
                    border-radius: 10px; padding: 16px 24px; margin-bottom: 20px;">
            <h3 style="margin:0">🏆 Best Model: {best_name}</h3>
            <p style="margin:4px 0 0 0; opacity:0.9">Average RMSE across 3 horizons: {avg_rmse:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Model comparison table
        st.markdown('<div class="section-title">All Model Results</div>', unsafe_allow_html=True)
        rows = []
        for name, res in report.get("all_results", {}).items():
            m = res.get("metrics", {})
            rows.append({
                "Model":     name,
                "RMSE t+1":  round(m.get("aqi_t1", m.get("AQI_t1", {})).get("rmse", 0), 2),
                "MAE t+1":   round(m.get("aqi_t1", m.get("AQI_t1", {})).get("mae",  0), 2),
                "R² t+1":    round(m.get("aqi_t1", m.get("AQI_t1", {})).get("r2",   0), 4),
                "RMSE t+2":  round(m.get("aqi_t2", m.get("AQI_t2", {})).get("rmse", 0), 2),
                "RMSE t+3":  round(m.get("aqi_t3", m.get("AQI_t3", {})).get("rmse", 0), 2),
                "Avg RMSE":  round(res.get("avg_rmse", 0), 2),
            })

        if rows:
            mdf = pd.DataFrame(rows).set_index("Model")
            st.dataframe(
                mdf.style
                   .highlight_min(subset=["Avg RMSE"], color="#2ecc71")
                   .format(precision=2),
                use_container_width=True,
            )

        # Model comparison bar chart
        st.markdown('<div class="section-title">RMSE Comparison</div>', unsafe_allow_html=True)
        if rows:
            names  = [r["Model"]    for r in rows]
            rmses  = [r["Avg RMSE"] for r in rows]
            colors = ["#2ecc71" if n == best_name else "#e67e22" for n in names]
            fig_cmp = go.Figure(go.Bar(
                x=names, y=rmses,
                marker_color=colors,
                text=[f"{r:.2f}" for r in rmses],
                textposition="outside",
            ))
            fig_cmp.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d0d0d",
                plot_bgcolor="#111",
                height=350,
                yaxis_title="Avg RMSE (lower = better)",
                showlegend=False,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("No training report found yet. Run the training pipeline first.")

    # ── SHAP feature importance ────────────────────────────────────────
    st.markdown('<div class="section-title">SHAP Feature Importance</div>', unsafe_allow_html=True)
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    shap_csvs  = []
    if os.path.exists(models_dir):
        shap_csvs = [f for f in os.listdir(models_dir) if f.endswith("_shap.csv")]

    if shap_csvs:
        shap_df = pd.read_csv(os.path.join(models_dir, shap_csvs[0]))
        shap_df = shap_df.sort_values("importance", ascending=False).head(15)
        fig_shap = go.Figure(go.Bar(
            x=shap_df["importance"][::-1].values,
            y=shap_df["feature"][::-1].values,
            orientation="h",
            marker_color="#e67e22",
        ))
        fig_shap.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d0d0d",
            plot_bgcolor="#111",
            height=450,
            xaxis_title="Mean |SHAP value|",
            title="Top 15 Features Driving AQI Prediction",
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP values will appear here after the training pipeline runs.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown("## ℹ️ About This Project")

    st.markdown(f"""
    <div style="background:#1a1a1a; border-radius:12px; padding:24px; margin-bottom:20px;
                border-left: 4px solid #e67e22;">
        <h3 style="margin:0 0 8px 0; color:#e67e22">🌫️ Karachi AQI Predictor</h3>
        <p style="color:#ccc; margin:0;">
        An end-to-end, fully automated Air Quality Index prediction system for Karachi, Pakistan.
        Predicts AQI for the next 3 days using a serverless MLOps stack with 6 competing models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🏗️ Architecture")
        arch = {
            "Data Sources":      "AQICN API + OpenWeatherMap",
            "Feature Store":     "Hopsworks (free tier)",
            "Model Registry":    "Hopsworks Model Registry",
            "CI/CD":             "GitHub Actions",
            "Orchestration":     "Apache Airflow",
            "ML Models":         "Ridge, RF, XGBoost, SARIMA, LSTM, Prophet",
            "Explainability":    "SHAP",
            "Dashboard":         "Streamlit",
            "API Backend":       "FastAPI",
            "Containerisation":  "Docker + docker-compose",
        }
        for k, v in arch.items():
            st.markdown(f"**{k}:** {v}")

    with c2:
        st.markdown("### 📅 Pipeline Schedule")
        st.markdown("""
        | Pipeline | Schedule |
        |---|---|
        | Feature collection | Every hour |
        | Model retraining | Every midnight |
        """)

        st.markdown("### 🔔 AQI Alert Levels")
        for lo, hi, clr, _, lbl, _, em in AQI_LEVELS:
            alert = " 🚨" if lo >= 151 else ""
            st.markdown(
                f"<span style='color:{clr}'>■</span> **{lo}–{hi}**: {lbl}{alert}",
                unsafe_allow_html=True
            )

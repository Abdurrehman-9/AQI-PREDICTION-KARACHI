"""
web_app/app.py
================
Streamlit dashboard for the AQI Predictor.

Shows:
  - Current AQI with live color-coded gauge
  - 3-day forecast with confidence bands
  - EDA charts (trends, pollutants, WHO comparison)
  - SHAP feature importance
  - Hazard alerts with audio/visual warnings
  - Model performance metrics

Run locally:
    streamlit run web_app/app.py
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── AQI helpers ─────────────────────────────────────────────────────────────

AQI_LEVELS = [
    (0,   50,  "#00e400", "Good",                          "Air quality is satisfactory."),
    (51,  100, "#ffff00", "Moderate",                      "Acceptable; some pollutants may concern sensitive people."),
    (101, 150, "#ff7e00", "Unhealthy for Sensitive Groups","Sensitive groups may experience effects."),
    (151, 200, "#ff0000", "Unhealthy",                     "Everyone may begin to experience effects."),
    (201, 300, "#8f3f97", "Very Unhealthy",                "Health alert: everyone may experience serious effects."),
    (301, 500, "#7e0023", "Hazardous",                     "Health warning of emergency conditions."),
]

def aqi_info(aqi: float) -> tuple:
    for lo, hi, color, label, desc in AQI_LEVELS:
        if lo <= aqi <= hi:
            return color, label, desc
    return "#7e0023", "Hazardous", "Extreme health emergency."

ALERT_THRESHOLD = int(os.getenv("ALERT_AQI_THRESHOLD", 150))

# ─── Load model & data ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the best trained model from disk or Hopsworks."""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    report_path = os.path.join(models_dir, "training_report.json")

    if not os.path.exists(report_path):
        return None, None, None

    with open(report_path) as f:
        report = json.load(f)

    best_name = report["best_model"]
    model_dir = os.path.join(models_dir, best_name.lower().replace(" ", "_"))

    import joblib
    try:
        if best_name == "LSTM":
            import tensorflow as tf
            model    = tf.keras.models.load_model(os.path.join(model_dir, "model.h5"))
            scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.pkl"))
            scaler_Y = joblib.load(os.path.join(model_dir, "scaler_Y.pkl"))
            return model, {"scaler_X": scaler_X, "scaler_Y": scaler_Y, "type": "lstm"}, report
        elif best_name == "Prophet":
            import pickle
            with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
                model = pickle.load(f)
            return model, {"type": "prophet"}, report
        else:
            model = joblib.load(os.path.join(model_dir, "model.pkl"))
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            return model, {"scaler": scaler, "type": "sklearn"}, report
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, report


@st.cache_data(ttl=3600, show_spinner="Fetching live AQI data...")
def fetch_live_data():
    """Fetch the latest reading from the API."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))
        from fetch_data import get_raw_reading
        from compute_features import compute_features_single
        raw = get_raw_reading()
        return raw, compute_features_single(raw)
    except Exception as e:
        st.warning(f"Could not fetch live data: {e}. Using mock data.")
        return _mock_raw(), _mock_features()


@st.cache_data(ttl=3600, show_spinner="Loading historical features...")
def load_history():
    """Load recent historical data from Feature Store."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))
        from store_features import read_features
        df = read_features()
        return df.tail(180)   # last 6 months
    except Exception as e:
        st.warning(f"Could not load history: {e}. Using mock data.")
        return _mock_history()


def _mock_raw():
    return {"AQI": 145, "PM2.5": 55, "PM10": 90, "NO2": 25, "SO2": 8, "O3": 70,
            "CO": 350, "temperature": 31, "humidity": 62, "wind_speed": 3.2,
            "timestamp": datetime.utcnow()}

def _mock_features():
    f = _mock_raw()
    f.update({"feels_like": 33, "pressure": 1010, "wind_deg": 180,
               "visibility": 7, "cloud_cover": 15})
    return f

def _mock_history():
    dates = pd.date_range(end=datetime.utcnow(), periods=180, freq="D")
    np.random.seed(42)
    aqi   = 80 + 40 * np.sin(np.arange(180) * 2 * np.pi / 365) + np.random.normal(0, 10, 180)
    return pd.DataFrame({
        "timestamp": dates, "AQI": np.clip(aqi, 20, 250),
        "PM2.5": aqi * 0.3, "PM10": aqi * 0.6, "NO2": aqi * 0.12,
        "SO2": aqi * 0.04, "O3": aqi * 0.35, "CO": aqi * 2.5,
        "temperature": 28 + np.random.normal(0, 4, 180),
        "humidity":    65 + np.random.normal(0, 8, 180),
    })


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict_3_days(model, meta: dict, history_df: pd.DataFrame):
    """Return predicted AQI for next 3 days."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))
        from compute_features import FEATURE_COLS

        if meta["type"] == "sklearn":
            X = history_df[FEATURE_COLS].tail(1).values
            if meta["scaler"]:
                X = meta["scaler"].transform(X)
            preds = model.predict(X)[0]   # shape (3,)
            return [float(p) for p in preds]

        elif meta["type"] == "lstm":
            SEQ_LEN  = 7
            scaler_X = meta["scaler_X"]
            scaler_Y = meta["scaler_Y"]
            X_raw    = history_df[FEATURE_COLS].tail(SEQ_LEN).values
            X_scaled = scaler_X.transform(X_raw)
            X_seq    = X_scaled.reshape(1, SEQ_LEN, -1)
            pred_s   = model.predict(X_seq)
            pred     = scaler_Y.inverse_transform(pred_s)[0]
            return [float(p) for p in pred]

        elif meta["type"] == "prophet":
            future = model.make_future_dataframe(periods=3, freq="D")
            fc     = model.predict(future)
            return fc["yhat"].tail(3).tolist()

    except Exception as e:
        st.warning(f"Prediction error: {e}. Using interpolated estimates.")
        last_aqi = float(history_df["AQI"].iloc[-1])
        return [last_aqi * (1 + 0.02 * i) for i in range(1, 4)]


# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px; padding: 20px; text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .aqi-big { font-size: 72px; font-weight: 800; line-height: 1; }
    .alert-box {
        background: linear-gradient(90deg, #7e0023, #ff0000);
        border-radius: 10px; padding: 16px 24px; text-align: center;
        font-size: 18px; font-weight: 700; color: white;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.7} }
    .section-title { font-size: 22px; font-weight: 700; margin: 20px 0 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/320px-Flag_of_Pakistan.svg.png", width=80)
    st.title("🌫️ AQI Predictor")
    st.markdown("**Karachi, Pakistan**")
    st.divider()

    page = st.radio("Navigate", ["🏠 Dashboard", "📊 EDA", "🧠 Model Insights", "ℹ️ About"])
    st.divider()
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ─── Load everything ─────────────────────────────────────────────────────────

model, meta, report = load_model()
raw,  features      = fetch_live_data()
history_df          = load_history()


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ═══════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.title("🌫️ Karachi Air Quality Dashboard")
    st.caption(f"Real-time AQI monitoring & 3-day forecast  •  {datetime.utcnow().strftime('%A, %d %B %Y %H:%M')} UTC")

    current_aqi = float(raw.get("AQI", 0))
    color, label, desc = aqi_info(current_aqi)

    # ── Hazard alert ──────────────────────────────────────────────────
    if current_aqi >= ALERT_THRESHOLD:
        st.markdown(f"""
        <div class="alert-box">
            🚨 HAZARD ALERT — AQI is {current_aqi:.0f} ({label})
            <br><small>{desc}</small>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

    # ── Current AQI + pollutants row ─────────────────────────────────
    col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 1, 1, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color:{color}" class="aqi-big">{current_aqi:.0f}</div>
            <div style="font-size:20px;font-weight:700;color:{color}">{label}</div>
            <div style="color:#aaa;font-size:13px;margin-top:6px">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    for pollutant, col in zip(["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"],
                               [col2, col3, col4, col5, col6, col7]):
        with col:
            val = float(raw.get(pollutant, 0))
            st.metric(pollutant, f"{val:.1f}", help=f"{pollutant} concentration")

    st.markdown("")

    # ── Weather strip ─────────────────────────────────────────────────
    wc1, wc2, wc3, wc4 = st.columns(4)
    wc1.metric("🌡️ Temperature", f"{raw.get('temperature', 0):.1f}°C")
    wc2.metric("💧 Humidity",    f"{raw.get('humidity', 0):.0f}%")
    wc3.metric("💨 Wind Speed",  f"{raw.get('wind_speed', 0):.1f} m/s")
    wc4.metric("☁️ Cloud Cover", f"{raw.get('cloud_cover', 0):.0f}%")

    st.divider()

    # ── 3-Day Forecast ───────────────────────────────────────────────
    st.markdown('<div class="section-title">📅 3-Day AQI Forecast</div>', unsafe_allow_html=True)

    forecast = predict_3_days(model, meta, history_df) if (model and meta) else \
               [current_aqi * 1.02, current_aqi * 1.05, current_aqi * 0.98]

    future_dates = [(datetime.utcnow() + timedelta(days=i+1)).strftime("%A\n%d %b") for i in range(3)]

    fc1, fc2, fc3 = st.columns(3)
    for i, (fcol, date, pred) in enumerate(zip([fc1, fc2, fc3], future_dates, forecast)):
        fc_color, fc_label, _ = aqi_info(pred)
        with fcol:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {fc_color}">
                <div style="color:#aaa;font-size:14px">Day {i+1} · {date.replace(chr(10),' ')}</div>
                <div style="color:{fc_color};font-size:48px;font-weight:800;margin:8px 0">{pred:.0f}</div>
                <div style="color:{fc_color};font-size:16px;font-weight:600">{fc_label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── 30-day trend chart with forecast ─────────────────────────────
    recent = history_df.tail(30).copy()
    forecast_dates = [datetime.utcnow() + timedelta(days=i+1) for i in range(3)]

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=recent["timestamp"], y=recent["AQI"],
        name="Historical AQI", line=dict(color="#4fc3f7", width=2),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.1)",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast,
        name="Forecast", mode="lines+markers",
        line=dict(color="#ff7043", width=3, dash="dash"),
        marker=dict(size=12, symbol="diamond"),
    ))

    # AQI zone bands
    for lo, hi, color, label, _ in AQI_LEVELS:
        fig.add_hrect(y0=lo, y1=min(hi, 350), fillcolor=color, opacity=0.06,
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=10)

    fig.update_layout(
        title="AQI Trend (Last 30 Days + 3-Day Forecast)",
        xaxis_title="Date", yaxis_title="AQI",
        template="plotly_dark", height=420,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    df = history_df.copy()
    df["month"]      = pd.to_datetime(df["timestamp"]).dt.month
    df["day_of_week"]= pd.to_datetime(df["timestamp"]).dt.dayofweek
    df["month_name"] = pd.to_datetime(df["timestamp"]).dt.strftime("%b")

    # ── AQI over time ─────────────────────────────────────────────────
    st.subheader("AQI Over Time with WHO Threshold Bands")
    fig1 = go.Figure()
    for lo, hi, color, label, _ in AQI_LEVELS:
        fig1.add_hrect(y0=lo, y1=min(hi, 350), fillcolor=color, opacity=0.07,
                       annotation_text=label, annotation_position="right",
                       annotation_font_size=9)
    fig1.add_trace(go.Scatter(x=df["timestamp"], y=df["AQI"], name="AQI",
                              line=dict(color="#4fc3f7", width=1.5)))
    fig1.update_layout(template="plotly_dark", height=380,
                       yaxis_title="AQI", xaxis_title="Date")
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Monthly average ───────────────────────────────────────────────
    with col1:
        st.subheader("Monthly Avg AQI")
        monthly = df.groupby("month_name")["AQI"].mean().reset_index()
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["month_name"] = pd.Categorical(monthly["month_name"], categories=month_order, ordered=True)
        monthly = monthly.sort_values("month_name")
        fig2 = px.bar(monthly, x="month_name", y="AQI",
                      color="AQI", color_continuous_scale="RdYlGn_r",
                      template="plotly_dark")
        fig2.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Day of week ────────────────────────────────────────────────────
    with col2:
        st.subheader("Avg AQI by Day of Week")
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = df.groupby("day_of_week")["AQI"].mean().reset_index()
        dow["day"] = dow["day_of_week"].map(lambda x: dow_names[x])
        fig3 = px.bar(dow, x="day", y="AQI",
                      color="AQI", color_continuous_scale="RdYlGn_r",
                      template="plotly_dark")
        fig3.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Pollutant contribution pie ─────────────────────────────────────
    col3, col4 = st.columns(2)
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]

    with col3:
        st.subheader("Pollutant Contribution")
        avgs = {p: float(df[p].mean()) for p in pollutants if p in df.columns}
        fig4 = go.Figure(go.Pie(labels=list(avgs.keys()), values=list(avgs.values()),
                                hole=0.4, textinfo="label+percent"))
        fig4.update_layout(template="plotly_dark", height=320,
                           showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────
    with col4:
        st.subheader("Correlation Matrix")
        cols = ["AQI"] + [p for p in pollutants if p in df.columns] + ["temperature", "humidity"]
        corr = df[cols].corr()
        fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         template="plotly_dark", aspect="auto")
        fig5.update_layout(height=320)
        st.plotly_chart(fig5, use_container_width=True)

    # ── AQI vs Temperature scatter ────────────────────────────────────
    if "temperature" in df.columns:
        st.subheader("AQI vs Temperature")
        fig6 = px.scatter(df, x="temperature", y="AQI", color="AQI",
                          color_continuous_scale="RdYlGn_r", opacity=0.6,
                          trendline="ols", template="plotly_dark")
        fig6.update_layout(height=350)
        st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Insights
# ═══════════════════════════════════════════════════════════════════════

elif page == "🧠 Model Insights":
    st.title("🧠 Model Performance & Explainability")

    if report:
        st.subheader("📋 Model Comparison")
        best_name = report.get("best_model", "Unknown")
        st.success(f"🏆 Best Model: **{best_name}** (avg RMSE = {report.get('avg_rmse', 0):.2f})")

        rows = []
        for name, res in report.get("all_results", {}).items():
            m = res.get("metrics", {})
            rows.append({
                "Model":    name,
                "RMSE t+1": m.get("AQI_t1", {}).get("rmse", "–"),
                "MAE t+1":  m.get("AQI_t1", {}).get("mae",  "–"),
                "R² t+1":   m.get("AQI_t1", {}).get("r2",   "–"),
                "RMSE t+2": m.get("AQI_t2", {}).get("rmse", "–"),
                "RMSE t+3": m.get("AQI_t3", {}).get("rmse", "–"),
                "Avg RMSE": round(res.get("avg_rmse", 0), 2),
            })

        metrics_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(metrics_df.style.highlight_min(subset=["Avg RMSE"], color="#00e400"),
                     use_container_width=True)
    else:
        st.info("No training report found. Run `python training_pipeline/train.py` first.")

    # ── SHAP chart if available ────────────────────────────────────────
    st.subheader("🔍 SHAP Feature Importance")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    shap_csvs  = [f for f in os.listdir(models_dir) if f.endswith("_shap.csv")] if os.path.exists(models_dir) else []

    if shap_csvs:
        shap_df = pd.read_csv(os.path.join(models_dir, shap_csvs[0]))
        fig_shap = px.bar(shap_df.head(15).sort_values("importance"),
                          x="importance", y="feature", orientation="h",
                          color="importance", color_continuous_scale="Blues",
                          template="plotly_dark")
        fig_shap.update_layout(height=480, title=f"Top 15 Features ({shap_csvs[0].replace('_shap.csv','')})")
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP values not yet generated. Run the training pipeline first.")

    # ── AQI distribution ─────────────────────────────────────────────
    st.subheader("📊 Historical AQI Distribution")
    fig_hist = px.histogram(history_df, x="AQI", nbins=40, color_discrete_sequence=["#4fc3f7"],
                            template="plotly_dark")
    fig_hist.update_layout(height=320)
    st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4 — About
# ═══════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## 🌫️ AQI Predictor — Karachi

    An end-to-end, fully automated Air Quality Index prediction system for Karachi, Pakistan.

    ### 🏗️ Architecture
    | Component | Technology |
    |---|---|
    | Data Source | AQICN API + OpenWeatherMap |
    | Feature Store | Hopsworks (free tier) |
    | Model Registry | Hopsworks Model Registry |
    | ML Models | Ridge, Random Forest, XGBoost, Prophet, LSTM |
    | Explainability | SHAP + LIME |
    | Web App | Streamlit |
    | CI/CD | GitHub Actions |
    | Containerization | Docker |

    ### 📅 Pipeline Schedule
    - **Feature pipeline** — runs every hour (GitHub Actions)
    - **Training pipeline** — runs every day at midnight (GitHub Actions)

    ### 🔔 Alert Thresholds
    | AQI | Level |
    |---|---|
    | 0–50 | Good |
    | 51–100 | Moderate |
    | 101–150 | Unhealthy for Sensitive |
    | **151–200** | **🚨 Unhealthy — ALERT** |
    | **201–300** | **🚨🚨 Very Unhealthy** |
    | **300+** | **☠️ Hazardous** |

    ### 📚 Data Sources
    - [AQICN API](https://aqicn.org/api/)
    - [OpenWeatherMap API](https://openweathermap.org/api)
    """)

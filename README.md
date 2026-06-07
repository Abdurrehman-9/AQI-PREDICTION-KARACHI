# Karachi AQI Predictor
### End-to-End Serverless Air Quality Forecasting System

> A production-grade MLOps pipeline that ingests live pollution and weather data every hour, engineers features, trains and benchmarks six forecasting models daily, and serves a 3-day AQI forecast for Karachi through an interactive dashboard — entirely automated, containerised, and deployed on free-tier cloud infrastructure.

---

## Overview

Air pollution is one of the most underreported public health crises in Pakistan. Karachi routinely records AQI values that exceed WHO safe limits, yet no accessible, real-time forecasting tool exists for the general public. This project addresses that gap by building a complete, automated machine learning system that predicts the Air Quality Index for the next three days using a combination of live pollutant readings, meteorological data, and a suite of models ranging from classical statistics to deep learning.

The system is built on a **serverless MLOps architecture**: all data is stored in a cloud feature store (Hopsworks), all pipelines run on a schedule without a dedicated server (GitHub Actions), all services are containerised for reproducibility (Docker), and the forecast is accessible through a public web dashboard (Streamlit Cloud) and a REST API (FastAPI). Nothing needs to be running on a personal machine for the system to continue operating.

---

## Key Highlights

- **Fully automated** — two pipelines run on schedule with zero manual intervention: feature collection every hour, model retraining every day
- **Six models benchmarked** — Ridge Regression, Random Forest, XGBoost, SARIMA, LSTM, and Prophet; the best performer by RMSE is automatically promoted to production
- **Correct problem framing** — AQI is treated as a continuous regression target throughout; regression losses (MSE) and regression metrics (RMSE, MAE, R²) are used exclusively
- **Explainable predictions** — SHAP values identify which features (PM2.5, humidity, wind speed, etc.) drive each forecast
- **Real-time hazard alerts** — dashboard fires a visual alert when predicted AQI crosses 150 (Unhealthy threshold)
- **Production patterns** — feature store versioning, model registry, CI/CD workflows, Airflow DAGs, and Dockerised multi-service deployment

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                            │
│                                                                  │
│        AQICN API                    OpenWeatherMap API           │
│   AQI · PM2.5 · PM10           Temperature · Humidity           │
│   NO2 · SO2 · O3 · CO           Wind · Pressure · Visibility    │
└───────────────────────┬──────────────────────────────────────────┘
                        │  raw readings · every hour
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FEATURE PIPELINE                            │
│                  (triggered every hour)                          │
│                                                                  │
│  fetch_data.py  →  compute_features.py  →  store_features.py    │
│                                                                  │
│  Features engineered:                                            │
│  · Time-based     hour · day_of_week · month · season           │
│  · Lag features   AQI_lag_1 · AQI_lag_2 · AQI_lag_3            │
│  · Change rate    AQI_diff (1-step difference)                   │
│  · Rolling stats  rolling_mean_3 · rolling_std_3                │
│  · Ratios         PM2.5/PM10 · NO2/SO2                          │
└───────────────────────┬──────────────────────────────────────────┘
                        │  structured feature rows
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                   HOPSWORKS  (free cloud tier)                   │
│                                                                  │
│    Feature Store                     Model Registry             │
│    ─────────────                     ──────────────             │
│    Versioned feature groups          Trained model artefacts    │
│    Online + offline access           Metrics linked to version  │
│    Full history of hourly rows       Best model served to API   │
└──────────┬───────────────────────────────────────┬──────────────┘
           │  read features                        │  register model
           ▼                                       │
┌─────────────────────────────┐                   │
│      TRAINING PIPELINE      │ ──────────────────┘
│   (triggered every day)     │
│                             │
│  1. Ridge Regression        │  Linear baseline · L2 regularised
│  2. Random Forest           │  Bagging · 300 trees · multi-output
│  3. XGBoost                 │  Gradient boosting · MSE objective
│  4. SARIMA (1,1,1)(1,1,1,7) │  Weekly seasonality · per-horizon
│  5. LSTM  (128 → 64 → 3)   │  14-day sequence · linear output
│  6. Prophet                 │  Trend + seasonality + regressors
│                             │
│  Evaluated by: RMSE · MAE · R²  (regression metrics)
│  Winner → saved to Hopsworks Model Registry                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      AUTOMATION LAYER                            │
│                                                                  │
│   GitHub Actions                    Apache Airflow              │
│   ──────────────                    ───────────────             │
│   Deploys code on every push        Manages pipeline execution  │
│   Triggers feature pipeline         Retries failed tasks        │
│   every hour via cron               Visual DAG dashboard        │
│   Triggers training pipeline        Full audit log per run      │
│   every midnight via cron           Dependency-aware scheduling │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        WEB LAYER                                 │
│              (all services run inside Docker)                    │
│                                                                  │
│   FastAPI  (api.py)               Streamlit  (app.py)           │
│   ─────────────────               ──────────────────            │
│   GET /current  → live AQI        Dashboard · live AQI gauge    │
│   GET /forecast → 3-day pred      3-day forecast cards          │
│   GET /history  → past N days     EDA · pollutant trends        │
│   Auto-generated /docs UI         SHAP feature importance       │
│   Consumed by Streamlit           Hazard alert system           │
│   + any external client           Model performance page        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **Data Ingestion** | AQICN API · OpenWeatherMap API | Live pollutant and meteorological data for Karachi |
| **Feature Store** | Hopsworks (free tier) | Versioned, cloud-hosted storage for all engineered features |
| **Model Registry** | Hopsworks Model Registry | Stores trained model artefacts with linked evaluation metrics |
| **CI/CD** | GitHub Actions | Deploys code and triggers both pipelines on cron schedules |
| **Pipeline Orchestration** | Apache Airflow | DAG-based scheduling, retry logic, and execution monitoring |
| **ML Models** | Scikit-learn · XGBoost · Statsmodels · TensorFlow · Prophet | Six-model regression benchmark suite |
| **Explainability** | SHAP | Feature importance for tree-based models |
| **Backend API** | FastAPI · Uvicorn | REST endpoints with auto-generated OpenAPI documentation |
| **Frontend Dashboard** | Streamlit | Interactive multi-page forecast and analysis dashboard |
| **Containerisation** | Docker · docker-compose | Four-service reproducible deployment |
| **Language** | Python 3.10 | Entire codebase |

---

## Machine Learning Models

**Problem type: Regression.** AQI is a continuous variable (e.g. 145.3, 89.0). All models output three continuous predictions: AQI at day+1, day+2, and day+3. Regression losses and metrics are used throughout — no classification metrics are applied.

| # | Model | Algorithm Family | Key Design Choices |
|---|---|---|---|
| 1 | Ridge Regression | Penalised linear | L2 regularisation · StandardScaler · MultiOutputRegressor wrapper · interpretable baseline |
| 2 | Random Forest | Bagging ensemble | 300 trees · native multi-output · no scaling needed · used for SHAP |
| 3 | XGBoost | Gradient boosting | `reg:squarederror` loss · 300 estimators · subsampling 0.8 · MultiOutputRegressor wrapper |
| 4 | SARIMA | Statistical time series | Order (1,1,1) · seasonal (1,1,1,7) · one model per horizon · statsmodels SARIMAX |
| 5 | LSTM | Recurrent deep learning | 14-day input window · 128→64 stacked layers · `linear` output activation · MSE loss · EarlyStopping |
| 6 | Prophet | Additive decomposition | Yearly + weekly seasonality · temperature / humidity / wind regressors · one model per horizon |

**Evaluation metrics:**

| Metric | Formula | Interpretation |
|---|---|---|
| RMSE | √(mean((ŷ − y)²)) | Penalises large errors more; same units as AQI |
| MAE | mean(\|ŷ − y\|) | Average error in AQI units; easy to communicate |
| R² | 1 − SS_res/SS_tot | Proportion of variance explained; 1.0 is perfect |

The model with the lowest **average RMSE across all three horizons** is automatically selected and registered as the production model.

---

## Repository Structure

```
aqi-predictor/
│
├── feature_pipeline/
│   ├── fetch_data.py           # AQICN + OpenWeatherMap API calls
│   ├── compute_features.py     # Feature engineering: lags, rolling, time, ratios
│   ├── store_features.py       # Hopsworks Feature Store read/write
│   └── run_pipeline.py         # Pipeline orchestrator (called by Actions + Airflow)
│
├── training_pipeline/
│   └── train.py                # All six models: train · evaluate · save · register
│
├── web_app/
│   ├── app.py                  # Streamlit dashboard (Dashboard · EDA · Insights · About)
│   └── api.py                  # FastAPI backend  (/current · /forecast · /history · /health)
│
├── airflow/
│   └── dags/
│       ├── feature_dag.py      # Airflow DAG: hourly feature pipeline
│       └── training_dag.py     # Airflow DAG: daily training pipeline
│
├── docker/
│   ├── Dockerfile.feature      # Container image for both pipelines
│   └── Dockerfile.webapp       # Container image for Streamlit + FastAPI
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # Cron: every hour  → run_pipeline.py
│       └── training_pipeline.yml   # Cron: every midnight → train.py
│
├── scripts/
│   └── backfill.py             # One-time historical data loader (2 years)
│
├── docker-compose.yml          # Four-service local stack
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Deployment

### Prerequisites

Sign up for the following free services and collect your API credentials before proceeding:

| Service | Purpose | Sign-up Link |
|---|---|---|
| AQICN | Real-time AQI and pollutant data | https://aqicn.org/api/ |
| OpenWeatherMap | Meteorological data | https://openweathermap.org/api |
| Hopsworks | Feature store and model registry | https://app.hopsworks.ai |
| GitHub | Code repository and CI/CD | https://github.com |
| Streamlit Cloud | Dashboard hosting | https://streamlit.io/cloud |

---

### Step 1 — Clone and Configure

```bash
git clone https://github.com/YOUR_USERNAME/aqi-predictor.git
cd aqi-predictor
cp .env.example .env
```

Open `.env` and fill in your four credentials:

```env
AQICN_TOKEN=your_token
OWM_API_KEY=your_key
HOPSWORKS_API_KEY=your_key
HOPSWORKS_PROJECT=aqi_karachi
```

---

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 — Backfill Historical Data (run once)

```bash
python scripts/backfill.py
```

Loads approximately two years of historical AQI and weather data into the Hopsworks Feature Store. Takes 5–10 minutes. Verify by opening your Hopsworks project and checking that the `aqi_features` feature group has populated rows.

---

### Step 4 — Train All Six Models

```bash
python training_pipeline/train.py
```

Trains and benchmarks all six models, prints a full comparison table of RMSE / MAE / R² per horizon, generates prediction plots and SHAP charts, and registers the winning model in Hopsworks. Runtime is approximately 15–30 minutes on a standard laptop (LSTM dominates training time).

---

### Step 5 — Run the Dashboard Locally

```bash
# Streamlit dashboard
streamlit run web_app/app.py
# → http://localhost:8501

# FastAPI backend (optional, separate terminal)
uvicorn web_app.api:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000/docs
```

---

### Step 6 — Enable Automated Pipelines (GitHub Actions)

Navigate to your GitHub repository → **Settings → Secrets and Variables → Actions** and add:

```
AQICN_TOKEN
OWM_API_KEY
HOPSWORKS_API_KEY
HOPSWORKS_PROJECT
```

The workflow files in `.github/workflows/` are already configured. Once secrets are added, GitHub will:
- Execute the **feature pipeline every hour** on GitHub-hosted runners
- Execute the **training pipeline every day at midnight UTC**

Monitor runs live under the **Actions** tab.

---

### Step 7 — Deploy Dashboard (Streamlit Cloud)

1. Go to https://share.streamlit.io → New App
2. Connect your GitHub repository
3. Set main file: `web_app/app.py`
4. Add the four environment variables under Advanced Settings
5. Deploy

The dashboard will be live at `https://your-username-aqi-predictor.streamlit.app`.

---

### Step 8 — Full Stack with Docker

```bash
docker-compose up --build
```

Starts four containers:

| Container | Service | URL |
|---|---|---|
| `aqi_webapp` | Streamlit dashboard | http://localhost:8501 |
| `aqi_api` | FastAPI backend | http://localhost:8000/docs |
| `aqi_airflow_webserver` | Airflow UI | http://localhost:8080 |
| `aqi_airflow_scheduler` | DAG scheduler | background |

```bash
docker-compose down   # stop all services
```

---

## AQI Reference

| AQI | Category | Health Guidance | Dashboard |
|---|---|---|---|
| 0 – 50 | Good | No risk | Green · no alert |
| 51 – 100 | Moderate | Acceptable for most | Yellow · no alert |
| 101 – 150 | Unhealthy for Sensitive Groups | Sensitive individuals limit outdoor time | Orange · no alert |
| 151 – 200 | Unhealthy | Everyone may experience effects | Red · 🚨 alert |
| 201 – 300 | Very Unhealthy | Health alert — avoid outdoor activity | Purple · 🚨🚨 high alert |
| 301+ | Hazardous | Emergency conditions | Dark red · ☠️ emergency |

---

## Resume Entry 

**Karachi AQI Predictor** · Personal Project · Python · MLOps

> Built a fully automated, end-to-end air quality forecasting system for Karachi using a serverless MLOps stack. Engineered two production pipelines — an hourly feature pipeline and a daily training pipeline — orchestrated via GitHub Actions (CI/CD) and Apache Airflow (DAG monitoring). Ingested live data from AQICN and OpenWeatherMap APIs, stored versioned features in a Hopsworks Feature Store, and benchmarked six regression models (Ridge, Random Forest, XGBoost, SARIMA, LSTM, Prophet) evaluated on RMSE, MAE, and R². Best model auto-registered to a Hopsworks Model Registry and served through a FastAPI backend and interactive Streamlit dashboard with SHAP-based explainability and real-time hazard alerts. Full stack containerised with Docker and deployed on free-tier cloud infrastructure.

**Skills demonstrated:** Python · Scikit-learn · XGBoost · TensorFlow · Prophet · Statsmodels · FastAPI · Streamlit · Apache Airflow · GitHub Actions · Docker · Hopsworks · Feature Engineering · Time Series Forecasting · SHAP · REST APIs · CI/CD · MLOps

---

*Built with Python 3.10 · Data: AQICN + OpenWeatherMap · Storage: Hopsworks · CI/CD: GitHub Actions · Orchestration: Apache Airflow · Serving: FastAPI + Streamlit · Containers: Docker*

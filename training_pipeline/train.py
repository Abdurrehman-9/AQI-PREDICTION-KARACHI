"""
training_pipeline/train.py
============================

PROBLEM TYPE: Regression
------------------------
AQI is a continuous variable (e.g. 145.3, 89.0, 212.7).
We predict its value 1, 2, and 3 days ahead.
All models here are regressors. All metrics are regression metrics.

Targets  : AQI_t1 (day+1), AQI_t2 (day+2), AQI_t3 (day+3)  — continuous floats
Metrics  : RMSE, MAE, R²   (regression metrics — correct for this problem)
Loss fns : MSE / Huber      (regression losses  — correct for this problem)

Models trained (in order of complexity):
  1. Ridge Regression        — linear baseline
  2. Random Forest           — tree ensemble
  3. XGBoost                 — gradient boosting
  4. SARIMA                  — statistical time-series (per-target)
  5. LSTM                    — deep learning, sequence model
  6. Prophet                 — Facebook's additive time-series model
"""

import os
import sys
import json
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))
from store_features import read_features
from compute_features import FEATURE_COLS, TARGET_COLS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# All three are regression metrics. Correct for continuous AQI targets.
#   RMSE — penalises large errors more (sensitive to outliers)
#   MAE  — average absolute error, easier to interpret in AQI units
#   R²   — proportion of variance explained (1.0 = perfect, 0 = predicts mean)
# ═══════════════════════════════════════════════════════════════════════════════

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """
    Compute RMSE, MAE, R² for a single target column.
    All three are valid only for continuous targets — which AQI is.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    if label:
        print(f"      {label:<12} → RMSE={rmse:6.2f}  MAE={mae:6.2f}  R²={r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_all_targets(Y_true: np.ndarray, Y_pred: np.ndarray, model_name: str) -> dict:
    """Evaluate all three horizon targets and return a metrics dict."""
    results = {}
    for i, target in enumerate(TARGET_COLS):
        results[target] = regression_metrics(Y_true[:, i], Y_pred[:, i], label=target)
    avg_rmse = float(np.mean([results[t]["rmse"] for t in TARGET_COLS]))
    results["avg_rmse"] = avg_rmse
    print(f"    -> {model_name} avg RMSE across 3 horizons: {avg_rmse:.2f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    print("Loading features from Hopsworks Feature Store...")
    df = read_features()
    df.dropna(subset=FEATURE_COLS + TARGET_COLS, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  {len(df)} rows  x  {len(FEATURE_COLS)} features  ->  {len(TARGET_COLS)} targets")
    return df


def split_data(df: pd.DataFrame, test_days: int = 90):
    """
    Time-aware split — never shuffle time-series data.
    Most recent test_days rows = test set. Everything before = train.
    """
    split_idx = len(df) - test_days
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
    Y_train = train_df[TARGET_COLS].values.astype(np.float32)
    Y_test  = test_df[TARGET_COLS].values.astype(np.float32)

    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    return X_train, X_test, Y_train, Y_test, train_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Ridge Regression  (linear baseline)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Ridge is a linear regressor with L2 regularisation.
# It maps: X (features) -> y_hat (continuous AQI)
# Loss used internally: MSE + lambda*||w||^2   (regression loss — correct)
# We standardise inputs because Ridge is scale-sensitive.

def train_ridge(X_train, X_test, Y_train, Y_test) -> dict:
    print("\n  -- Model 1: Ridge Regression --")
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    # MultiOutputRegressor wraps Ridge to handle 3 targets simultaneously
    model  = MultiOutputRegressor(Ridge(alpha=1.0))
    model.fit(Xtr_s, Y_train)

    Y_pred  = model.predict(Xte_s)
    metrics = evaluate_all_targets(Y_test, Y_pred, "Ridge")

    return {
        "model":    model,
        "scaler":   scaler,
        "metrics":  metrics,
        "avg_rmse": metrics["avg_rmse"],
        "type":     "sklearn",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Random Forest  (tree ensemble / bagging)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Builds many decision trees on random subsets of data+features, averages them.
# Handles non-linearity and feature interactions well.
# Loss used internally: MSE at each split   (regression — correct)
# No feature scaling needed (trees are scale-invariant).

def train_random_forest(X_train, X_test, Y_train, Y_test) -> dict:
    print("\n  -- Model 2: Random Forest --")
    model = RandomForestRegressor(
        n_estimators      = 300,
        max_depth         = None,
        min_samples_split = 5,
        random_state      = 42,
        n_jobs            = -1,
    )
    model.fit(X_train, Y_train)   # RF natively supports multi-output

    Y_pred  = model.predict(X_test)
    metrics = evaluate_all_targets(Y_test, Y_pred, "RandomForest")

    return {
        "model":    model,
        "scaler":   None,
        "metrics":  metrics,
        "avg_rmse": metrics["avg_rmse"],
        "type":     "sklearn",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — XGBoost  (gradient boosting)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Builds trees sequentially; each tree corrects the residuals of the previous.
# Typically the most accurate tabular model.
# Loss used internally: MSE (reg:squarederror)   (regression — correct)

def train_xgboost(X_train, X_test, Y_train, Y_test) -> dict:
    print("\n  -- Model 3: XGBoost --")
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  XGBoost not installed. Run: pip install xgboost")
        return None

    # XGBoost does not natively support multi-output, so we wrap it
    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators     = 300,
            learning_rate    = 0.05,
            max_depth        = 6,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            objective        = "reg:squarederror",   # MSE loss — regression
            random_state     = 42,
            verbosity        = 0,
        )
    )
    model.fit(X_train, Y_train)

    Y_pred  = model.predict(X_test)
    metrics = evaluate_all_targets(Y_test, Y_pred, "XGBoost")

    return {
        "model":    model,
        "scaler":   None,
        "metrics":  metrics,
        "avg_rmse": metrics["avg_rmse"],
        "type":     "sklearn",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — SARIMA  (Seasonal ARIMA)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Statistical time-series model.
# ARIMA = AutoRegressive Integrated Moving Average
# SARIMA adds Seasonal terms (S) — important for AQI (weekly + annual cycles).
#
# Trained separately for each target horizon (t+1, t+2, t+3).
# Uses only the AQI series itself (univariate).
# Produces continuous float forecasts — regression.
#
# order=(p,d,q):
#   p=1  autoregressive:   uses 1 past value
#   d=1  differencing:     subtracts previous value to make series stationary
#   q=1  moving average:   uses 1 past forecast error
# seasonal_order=(P,D,Q,s):
#   s=7  weekly season (7 days)
#   P,D,Q mirror p,d,q but for the seasonal component

def train_sarima(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    print("\n  -- Model 4: SARIMA --")
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("  statsmodels not installed. Run: pip install statsmodels")
        return None

    aqi_train = train_df["AQI"].values.astype(float)
    n_test    = len(test_df)

    order          = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)

    sarima_models = {}
    Y_pred_all    = np.zeros((n_test, 3))
    Y_true_all    = test_df[TARGET_COLS].values.astype(float)

    for i, target in enumerate(TARGET_COLS):
        horizon = i + 1
        print(f"    Training SARIMA for {target} (horizon={horizon})...", end=" ")

        try:
            model  = SARIMAX(
                aqi_train,
                order          = order,
                seasonal_order = seasonal_order,
                enforce_stationarity  = False,
                enforce_invertibility = False,
            )
            fitted   = model.fit(disp=False, maxiter=100)
            forecast = fitted.forecast(steps=n_test + horizon)
            Y_pred_all[:, i] = forecast[horizon - 1 : n_test + horizon - 1]
            sarima_models[target] = fitted
            print("done")
        except Exception as e:
            print(f"failed: {e}")
            Y_pred_all[:, i] = aqi_train[-1]   # naive fallback

    metrics = evaluate_all_targets(Y_true_all, Y_pred_all, "SARIMA")

    return {
        "model":    sarima_models,
        "scaler":   None,
        "metrics":  metrics,
        "avg_rmse": metrics["avg_rmse"],
        "type":     "sarima",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 5 — LSTM  (Long Short-Term Memory)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Recurrent deep learning model that learns from sequences.
# Input : sliding window of last SEQ_LEN timesteps of all features
# Output: [AQI_t1, AQI_t2, AQI_t3]  — 3 continuous float values
#
# Loss function   : MSE (mean_squared_error)  — regression loss, correct
# Output activation: linear                  — regression output, correct
#
# NOTE: If this were a classification problem (predicting AQI category),
#   the output layer would use softmax activation and the loss would be
#   categorical_crossentropy. We do NOT use those here.

def train_lstm(X_train, X_test, Y_train, Y_test) -> dict:
    print("\n  -- Model 5: LSTM --")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        print("  TensorFlow not installed. Run: pip install tensorflow")
        return None

    SEQ_LEN = 14   # use last 14 days as context window

    # Scale features and targets to zero-mean, unit-variance
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_all   = np.vstack([X_train, X_test])
    Y_all   = np.vstack([Y_train, Y_test])
    X_all_s = scaler_X.fit_transform(X_all)
    Y_all_s = scaler_Y.fit_transform(Y_all)

    X_tr_s = X_all_s[:len(X_train)]
    X_te_s = X_all_s[len(X_train):]
    Y_tr_s = Y_all_s[:len(Y_train)]
    Y_te_s = Y_all_s[len(Y_train):]

    def make_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int):
        Xs, Ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            Ys.append(Y[i + seq_len])
        return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)

    X_tr_seq, Y_tr_seq = make_sequences(X_tr_s, Y_tr_s, SEQ_LEN)
    X_te_seq, Y_te_seq = make_sequences(X_te_s, Y_te_s, SEQ_LEN)

    n_features = X_tr_seq.shape[2]
    n_outputs  = Y_tr_seq.shape[1]   # = 3

    # Architecture: two stacked LSTM layers -> Dense regression head
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, n_features)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(n_outputs, activation="linear"),   # linear = continuous regression output
    ])

    # MSE loss for regression. Would be categorical_crossentropy for classification.
    model.compile(
        optimizer = Adam(learning_rate=1e-3),
        loss      = "mse",
        metrics   = ["mae"],
    )

    print(f"    Input shape: ({SEQ_LEN}, {n_features})  ->  Output: {n_outputs} continuous values")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
    ]

    history = model.fit(
        X_tr_seq, Y_tr_seq,
        epochs           = 150,
        batch_size       = 32,
        validation_split = 0.1,
        callbacks        = callbacks,
        verbose          = 0,
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"    Stopped at epoch {best_epoch}  (best val_loss = {min(history.history['val_loss']):.4f})")

    # Inverse-scale predictions back to original AQI range
    Y_pred_s = model.predict(X_te_seq, verbose=0)
    Y_pred   = scaler_Y.inverse_transform(Y_pred_s)
    Y_true   = scaler_Y.inverse_transform(Y_te_seq)

    metrics = evaluate_all_targets(Y_true, Y_pred, "LSTM")
    _plot_lstm_loss(history, OUTPUT_DIR)

    return {
        "model":    model,
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "seq_len":  SEQ_LEN,
        "scaler":   None,
        "metrics":  metrics,
        "avg_rmse": metrics["avg_rmse"],
        "type":     "lstm",
    }


def _plot_lstm_loss(history, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"],     label="Train MSE loss", color="steelblue")
    ax.plot(history.history["val_loss"], label="Val MSE loss",   color="crimson", linestyle="--")
    ax.set_title("LSTM — MSE Loss During Training (Regression)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "lstm_loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Loss curve -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 6 — Prophet
# ═══════════════════════════════════════════════════════════════════════════════
#
# Facebook/Meta's additive regression model for time series.
# Decomposes the series into: trend + yearly seasonality + weekly seasonality
#   + weather regressors.
# All outputs are continuous floats — regression.
# Trained separately for each target horizon.

def train_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    print("\n  -- Model 6: Prophet --")
    try:
        from prophet import Prophet
    except ImportError:
        print("  Prophet not installed. Run: pip install prophet")
        return None

    REGRESSORS = [c for c in ["temperature", "humidity", "wind_speed"] if c in train_df.columns]

    n_test     = len(test_df)
    Y_pred_all = np.zeros((n_test, 3))
    Y_true_all = test_df[TARGET_COLS].values.astype(float)
    prophet_models = {}

    for i, target in enumerate(TARGET_COLS):
        horizon = i + 1
        print(f"    Training Prophet for {target} (horizon={horizon})...", end=" ")

        # Prophet requires columns named 'ds' (datetime) and 'y' (value to forecast)
        train_p = train_df[["timestamp"] + REGRESSORS].copy()
        train_p.rename(columns={"timestamp": "ds"}, inplace=True)
        train_p["y"] = train_df["AQI"].values

        m = Prophet(
            yearly_seasonality      = True,
            weekly_seasonality      = True,
            daily_seasonality       = False,
            changepoint_prior_scale = 0.05,
        )
        for reg in REGRESSORS:
            m.add_regressor(reg)

        m.fit(train_p)

        future = test_df[["timestamp"] + REGRESSORS].copy()
        future.rename(columns={"timestamp": "ds"}, inplace=True)
        forecast = m.predict(future)

        # yhat = continuous AQI prediction
        Y_pred_all[:, i]  = forecast["yhat"].values
        prophet_models[target] = m
        print("done")

    metrics = evaluate_all_targets(Y_true_all, Y_pred_all, "Prophet")

    return {
        "model":    prophet_models,
        "scaler":   None,
        "metrics":  metrics,
        "avg_rmse": metrics["avg_rmse"],
        "type":     "prophet",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE & REGISTER
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(name: str, result: dict) -> str:
    slug      = name.lower().replace(" ", "_")
    model_dir = os.path.join(OUTPUT_DIR, slug)
    os.makedirs(model_dir, exist_ok=True)
    mtype = result["type"]

    if mtype == "lstm":
        result["model"].save(os.path.join(model_dir, "model.keras"))
        joblib.dump(result["scaler_X"], os.path.join(model_dir, "scaler_X.pkl"))
        joblib.dump(result["scaler_Y"], os.path.join(model_dir, "scaler_Y.pkl"))
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({"seq_len": result["seq_len"]}, f)

    elif mtype in ("sarima", "prophet"):
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump(result["model"], f)

    else:   # sklearn
        joblib.dump(result["model"], os.path.join(model_dir, "model.pkl"))
        if result.get("scaler"):
            joblib.dump(result["scaler"], os.path.join(model_dir, "scaler.pkl"))

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(result["metrics"], f, indent=2)

    print(f"  Saved {name} -> {model_dir}")
    return model_dir


def register_in_hopsworks(model_name: str, model_dir: str, metrics: dict):
    try:
        import hopsworks
        from dotenv import load_dotenv
        load_dotenv()

        project = hopsworks.login(
            api_key_value = os.getenv("HOPSWORKS_API_KEY"),
            project       = os.getenv("HOPSWORKS_PROJECT", "aqi_karachi"),
        )
        mr    = project.get_model_registry()
        model = mr.python.create_model(
            name        = f"aqi_{model_name.lower().replace(' ', '_')}",
            description = f"AQI 3-day regression model: {model_name}",
            metrics = {
                "avg_rmse": round(metrics.get("avg_rmse", 0), 3),
                "rmse_t1":  round(metrics.get("AQI_t1", {}).get("rmse", 0), 3),
                "r2_t1":    round(metrics.get("AQI_t1", {}).get("r2",   0), 4),
            },
        )
        model.save(model_dir)
        print(f"  Registered '{model_name}' in Hopsworks Model Registry")
    except Exception as e:
        print(f"  Hopsworks registration skipped: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_predictions(test_df: pd.DataFrame, Y_test: np.ndarray,
                     Y_pred: np.ndarray, model_name: str):
    dates = test_df["timestamp"].values[-len(Y_test):]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for i, target in enumerate(TARGET_COLS):
        axes[i].plot(dates, Y_test[:, i], label="Actual AQI",  color="steelblue", linewidth=1.5)
        axes[i].plot(dates, Y_pred[:, i], label="Predicted",   color="crimson",   linewidth=1.5, linestyle="--")
        rmse = float(np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i])))
        axes[i].set_title(f"{target}  |  RMSE = {rmse:.2f}", fontsize=10)
        axes[i].set_ylabel("AQI")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} — Regression: Predicted vs Actual AQI (Test Set)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Prediction plot -> {path}")


def plot_model_comparison(all_results: dict):
    names  = list(all_results.keys())
    rmses  = [all_results[n]["avg_rmse"] for n in names]
    colors = ["#00e676" if r == min(rmses) else "#4fc3f7" for r in rmses]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, rmses, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    ax.set_ylabel("Avg RMSE across 3 horizons (lower = better)")
    ax.set_title("Regression Model Comparison — AQI Forecasting")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Model comparison chart -> {path}")


def generate_shap(model, X_test: np.ndarray, model_name: str):
    try:
        import shap
        print(f"\n  Generating SHAP values for {model_name}...")
        base_model  = model.estimators_[0] if hasattr(model, "estimators_") else model
        explainer   = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_test[:200])
        mean_shap   = np.abs(shap_values).mean(axis=0)
        shap_df     = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": mean_shap,
        }).sort_values("importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(shap_df["feature"][::-1], shap_df["importance"][::-1], color="steelblue")
        ax.set_title(f"SHAP Feature Importance — {model_name} (AQI_t1)")
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_shap.png")
        plt.savefig(path, dpi=150)
        plt.close()
        shap_df.to_csv(os.path.join(OUTPUT_DIR, f"{model_name.lower()}_shap.csv"), index=False)
        print(f"  SHAP plot -> {path}")
    except Exception as e:
        print(f"  SHAP skipped: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AQI Training Pipeline  |  Problem type: REGRESSION")
    print("Targets : AQI_t1, AQI_t2, AQI_t3  (continuous floats)")
    print("Metrics : RMSE, MAE, R2  (regression metrics)")
    print("=" * 60)

    df = load_data()
    X_train, X_test, Y_train, Y_test, train_df, test_df = split_data(df)

    all_results: dict = {}

    r = train_ridge(X_train, X_test, Y_train, Y_test)
    if r: all_results["Ridge"] = r

    r = train_random_forest(X_train, X_test, Y_train, Y_test)
    if r: all_results["RandomForest"] = r

    r = train_xgboost(X_train, X_test, Y_train, Y_test)
    if r: all_results["XGBoost"] = r

    r = train_sarima(train_df, test_df)
    if r: all_results["SARIMA"] = r

    r = train_lstm(X_train, X_test, Y_train, Y_test)
    if r: all_results["LSTM"] = r

    r = train_prophet(train_df, test_df)
    if r: all_results["Prophet"] = r

    # ── Results table ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS  (all metrics are regression metrics)")
    print(f"  {'Model':<14} {'RMSE t+1':>9} {'RMSE t+2':>9} {'RMSE t+3':>9} {'Avg RMSE':>10} {'R2 t+1':>8}")
    print("  " + "-" * 57)
    for name, res in all_results.items():
        m   = res["metrics"]
        r1  = m["AQI_t1"]["rmse"]
        r2  = m["AQI_t2"]["rmse"]
        r3  = m["AQI_t3"]["rmse"]
        r2v = m["AQI_t1"]["r2"]
        avg = res["avg_rmse"]
        print(f"  {name:<14} {r1:>9.2f} {r2:>9.2f} {r3:>9.2f} {avg:>10.2f} {r2v:>8.4f}")

    best_name = min(all_results, key=lambda k: all_results[k]["avg_rmse"])
    best      = all_results[best_name]
    print(f"\n  Best: {best_name}  (avg RMSE = {best['avg_rmse']:.2f})")

    # ── Plots ──────────────────────────────────────────────────────────
    plot_model_comparison(all_results)

    if best["type"] == "sklearn":
        X_in   = X_test
        scaler = best.get("scaler")
        if scaler:
            X_in = scaler.transform(X_test)
        Y_pred = best["model"].predict(X_in)
        plot_predictions(test_df, Y_test, Y_pred, best_name)

    if best_name in ("RandomForest", "XGBoost"):
        generate_shap(best["model"], X_test, best_name)

    # ── Save all models ────────────────────────────────────────────────
    print("\nSaving all models...")
    model_dirs = {}
    for name, res in all_results.items():
        model_dirs[name] = save_model(name, res)

    print(f"\nRegistering best model ({best_name}) in Hopsworks...")
    register_in_hopsworks(best_name, model_dirs[best_name], best["metrics"])

    # ── Training report ────────────────────────────────────────────────
    report = {
        "trained_at":   datetime.utcnow().isoformat(),
        "problem_type": "regression",
        "targets":      TARGET_COLS,
        "metrics_used": ["RMSE", "MAE", "R2"],
        "best_model":   best_name,
        "avg_rmse":     round(best["avg_rmse"], 3),
        "all_results": {
            name: {
                "avg_rmse": round(res["avg_rmse"], 3),
                "metrics":  res["metrics"],
            }
            for name, res in all_results.items()
        },
    }
    report_path = os.path.join(OUTPUT_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTraining complete!  Report -> {report_path}\n")


if __name__ == "__main__":
    main()

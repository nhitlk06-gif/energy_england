"""Interactive dashboard for the UK electricity demand forecasting service.

Run with:

    streamlit run app/streamlit_app.py

By default this talks directly to the trained models on disk (no API
server required). If ``ENERGYFORECAST_API_URL`` is set, it instead calls
the FastAPI service at that URL — handy when the API is deployed
separately.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

API_URL = os.environ.get("ENERGYFORECAST_API_URL", "").rstrip("/")

st.set_page_config(page_title="UK Electricity Demand Forecast", page_icon="⚡", layout="wide")
st.title("⚡ UK National Grid Electricity Demand Forecast")
st.caption(
    "National Demand (ND, MW) — England, Wales & Scotland transmission system · "
    "trained on National Grid ESO half-hourly settlement data, 2020-2025"
)

DISPLAY_NAMES = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest Regressor",
    "xgboost": "XGBoost Regressor",
    "lightgbm": "LightGBM Regressor",
}


@st.cache_resource
def _local_artifacts():
    from energyforecast.forecast import load_artifacts

    return load_artifacts()


def using_api() -> bool:
    return bool(API_URL)


def api_get(path: str, **params):
    import requests

    resp = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_models():
    if using_api():
        return api_get("/models")
    artifacts = _local_artifacts()
    metrics_by_name = {}
    if artifacts.metrics_summary:
        metrics_by_name = {row["Model Name"]: row for row in artifacts.metrics_summary}
    return [
        {"key": k, "display_name": DISPLAY_NAMES.get(k, k), "metrics": metrics_by_name.get(DISPLAY_NAMES.get(k, k))}
        for k in artifacts.available_models()
    ]


def get_history(periods: int) -> pd.DataFrame:
    if using_api():
        rows = api_get("/history", periods=periods)
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.set_index("datetime")
    artifacts = _local_artifacts()
    from energyforecast import config

    tail = artifacts.history[config.PRIMARY_TARGET].tail(periods).rename("nd_mw").to_frame()
    return tail


def get_forecast(steps: int, model_key: str | None) -> pd.DataFrame:
    if using_api():
        payload = api_get("/forecast", steps=steps, model=model_key)
        df = pd.DataFrame(payload["forecast"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.set_index("datetime"), payload["model_used"]
    from energyforecast.forecast import forecast_horizon

    artifacts = _local_artifacts()
    result = forecast_horizon(artifacts, n_steps=steps, model_key=model_key)
    return result, result.attrs.get("model_used", model_key or "auto")


try:
    model_infos = get_models()
except Exception as exc:  # noqa: BLE001
    st.error(
        "Could not load trained models. Run the training pipeline first "
        "(`python -m energyforecast.pipeline` or `make pipeline`).\n\n"
        f"Details: {exc}"
    )
    st.stop()

if not model_infos:
    st.warning("No trained models found yet. Run the training pipeline first.")
    st.stop()

with st.sidebar:
    st.header("Forecast settings")
    model_options = {m["display_name"]: m["key"] for m in model_infos}
    chosen_display = st.selectbox("Model", ["Best available"] + list(model_options.keys()))
    chosen_key = None if chosen_display == "Best available" else model_options[chosen_display]

    horizon_choice = st.select_slider(
        "Forecast horizon", options=[6, 12, 24, 48, 96, 144, 336], value=48,
        help="Number of 30-minute settlement periods to forecast ahead",
    )
    history_periods = st.slider("History periods to show", 48, 2000, 336, step=48)

    st.divider()
    st.subheader("Model leaderboard (2025 held-out test)")
    for m in model_infos:
        if m.get("metrics"):
            met = m["metrics"]
            st.metric(
                m["display_name"],
                f"RMSE {met.get('RMSE', float('nan')):.1f} MW",
                help=f"R2={met.get('R2', float('nan')):.4f} | MAE={met.get('MAE', float('nan')):.1f} MW | MAPE={met.get('MAPE (%)', float('nan')):.2f}%",
            )

col1, col2 = st.columns([3, 1])

with st.spinner("Loading history and computing forecast..."):
    history_df = get_history(history_periods)
    forecast_df, model_used = get_forecast(horizon_choice, chosen_key)

with col1:
    st.subheader(f"National Demand: recent history + {horizon_choice}-period forecast ({model_used})")
    plot_df = pd.DataFrame(index=history_df.index.union(forecast_df.index))
    plot_df["Observed (MW)"] = history_df.iloc[:, 0]
    plot_df["Forecast (MW)"] = forecast_df["forecast_mw"]
    st.line_chart(plot_df)

with col2:
    st.subheader("Forecast summary")
    st.metric("Model used", model_used)
    st.metric("Horizon", f"{horizon_choice} periods ({horizon_choice * 0.5:.0f} h)")
    st.metric("Peak forecast", f"{forecast_df['forecast_mw'].max():,.0f} MW")
    st.metric("Trough forecast", f"{forecast_df['forecast_mw'].min():,.0f} MW")

st.subheader("Forecast table")
st.dataframe(forecast_df.rename(columns={"forecast_mw": "Forecast ND (MW)"}), width="stretch")

st.caption(
    "Research/demo tool. Forecasts beyond the last observed settlement period are produced "
    "recursively: each 30-minute-ahead prediction is fed back in as if observed to build the "
    "lag/rolling features for the next step, and exogenous columns (e.g. embedded renewable "
    "generation) are approximated with same-period-yesterday persistence. This is not an "
    "official National Grid ESO forecast."
)

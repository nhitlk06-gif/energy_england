"""FastAPI service exposing the trained UK electricity demand forecasting
models.

Run locally with:

    uvicorn app.api:app --reload --port 8000

Then browse the interactive docs at http://localhost:8000/docs

Endpoints
---------
GET  /health              liveness check
GET  /models               list available trained models + held-out test metrics
GET  /forecast              recursively forecast National Demand (MW) N steps ahead
GET  /history               recent observed National Demand history
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Make the src/ package importable when running this file directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from energyforecast import config  # noqa: E402
from energyforecast.forecast import (  # noqa: E402
    ForecastArtifacts,
    evaluate_on_test_set,
    forecast_horizon,
    load_artifacts,
)

app = FastAPI(
    title="UK Electricity Demand Forecast API",
    description=(
        "Forecasts National Demand (ND, in MW) for the England & Wales / GB "
        "transmission system, using models trained on six years (2020-2025) "
        "of National Grid ESO half-hourly settlement data."
    ),
    version="1.0.0",
)

_artifacts: Optional[ForecastArtifacts] = None


def get_artifacts() -> ForecastArtifacts:
    global _artifacts
    if _artifacts is None:
        try:
            _artifacts = load_artifacts()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _artifacts


class ForecastPoint(BaseModel):
    datetime: str
    forecast_mw: float


class ForecastResponse(BaseModel):
    model_used: str
    n_steps: int
    period_minutes: int = 30
    forecast: List[ForecastPoint]


class HistoryPoint(BaseModel):
    datetime: str
    nd_mw: float


class ModelInfo(BaseModel):
    key: str
    display_name: str
    metrics: Optional[dict] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models", response_model=List[ModelInfo])
def list_models() -> List[ModelInfo]:
    artifacts = get_artifacts()
    display_names = {
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest Regressor",
        "xgboost": "XGBoost Regressor",
        "lightgbm": "LightGBM Regressor",
    }
    metrics_by_name = {}
    if artifacts.metrics_summary:
        metrics_by_name = {row["Model Name"]: row for row in artifacts.metrics_summary}

    infos = []
    for key in artifacts.available_models():
        display = display_names.get(key, key)
        infos.append(ModelInfo(key=key, display_name=display, metrics=metrics_by_name.get(display)))
    return infos


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    steps: int = Query(48, ge=1, le=2000, description="Number of 30-minute settlement periods to forecast"),
    model: Optional[str] = Query(
        None, description="Model key: linear_regression | random_forest | xgboost | lightgbm. Default: best model."
    ),
) -> ForecastResponse:
    artifacts = get_artifacts()
    try:
        result = forecast_horizon(artifacts, n_steps=steps, model_key=model)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    points = [
        ForecastPoint(datetime=idx.isoformat(), forecast_mw=round(float(val), 2))
        for idx, val in result["forecast_mw"].items()
    ]
    return ForecastResponse(model_used=result.attrs.get("model_used", model or "auto"), n_steps=steps, forecast=points)


@app.get("/history", response_model=List[HistoryPoint])
def get_history(
    periods: int = Query(96, ge=1, le=5000, description="Number of most recent 30-minute periods to return"),
) -> List[HistoryPoint]:
    artifacts = get_artifacts()
    tail = artifacts.history[config.PRIMARY_TARGET].tail(periods)
    return [HistoryPoint(datetime=idx.isoformat(), nd_mw=round(float(val), 2)) for idx, val in tail.items()]


@app.get("/evaluate")
def get_evaluation(model: Optional[str] = Query(None)) -> dict:
    artifacts = get_artifacts()
    try:
        return evaluate_on_test_set(artifacts, model_key=model)
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

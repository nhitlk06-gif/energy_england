"""Inference utilities: load trained artifacts and produce forecasts.

This module is what the FastAPI service (``app/api.py``) calls. It supports
two things:

1. ``load_artifacts`` — load the trained model + the list of feature
   columns it expects, plus enough recent history to build features from.
2. ``forecast_horizon`` — recursively forecast ``ND`` (National Demand,
   in MW) ``n_steps`` settlement periods (each 30 minutes) into the future
   beyond the end of the available history, feeding each prediction back
   in as if it were an observation for the next step's lag/rolling
   features (a standard recursive/direct-multistep strategy for
   autoregressive feature sets).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from . import config
from .features import build_feature_row_for_inference, load_clean_series
from .splits import load_feature_matrix

logger = logging.getLogger(__name__)

MODEL_DISPLAY_NAMES = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest Regressor",
    "xgboost": "XGBoost Regressor",
    "lightgbm": "LightGBM Regressor",
}


@dataclass
class ForecastArtifacts:
    models: Dict[str, object]
    feature_columns: List[str]
    history: pd.DataFrame  # tail of the cleaned series, target + exogenous cols
    metrics_summary: Optional[List[dict]]

    def available_models(self) -> List[str]:
        return list(self.models.keys())


def _load_history(models_dir: Path) -> pd.DataFrame:
    """Prefer the full feature matrix (has everything precomputed); fall
    back to the cleaned series if the feature matrix isn't available.
    """
    if config.FEATURES_CSV.exists():
        df = load_feature_matrix(config.FEATURES_CSV)
        return df
    if config.CLEANED_CSV.exists():
        return load_clean_series(config.CLEANED_CSV)
    raise FileNotFoundError(
        "Neither the feature matrix nor the cleaned CSV were found. "
        "Run the cleaning and feature pipelines first."
    )


def load_artifacts(models_dir: Optional[Path] = None) -> ForecastArtifacts:
    models_dir = Path(models_dir or config.MODELS_DIR)

    fitted = {}
    for key, filename in config.MODEL_FILENAMES.items():
        path = models_dir / filename
        if path.exists():
            fitted[key] = joblib.load(path)

    feature_columns_path = models_dir / "feature_columns.json"
    if feature_columns_path.exists():
        feature_columns = json.loads(feature_columns_path.read_text())
    else:
        feature_columns = [c for c in _load_history(models_dir).columns if c not in config.TARGET_COLUMNS]

    metrics_summary = None
    metrics_path = models_dir / "metrics_summary.json"
    if metrics_path.exists():
        metrics_summary = json.loads(metrics_path.read_text())

    history = _load_history(models_dir)

    return ForecastArtifacts(
        models=fitted,
        feature_columns=feature_columns,
        history=history,
        metrics_summary=metrics_summary,
    )


def _select_model(artifacts: ForecastArtifacts, model_key: Optional[str]):
    if not artifacts.models:
        raise RuntimeError("No trained models are available. Run the training pipeline first.")

    if model_key is None:
        # Default to the best model on the held-out test metrics if we have
        # them, otherwise XGBoost, otherwise whatever is available.
        if artifacts.metrics_summary:
            best_name = artifacts.metrics_summary[0]["Model Name"]
            for key, display in MODEL_DISPLAY_NAMES.items():
                if display == best_name and key in artifacts.models:
                    return key, artifacts.models[key]
        for preferred in ("xgboost", "lightgbm", "random_forest", "linear_regression"):
            if preferred in artifacts.models:
                return preferred, artifacts.models[preferred]
        key = next(iter(artifacts.models))
        return key, artifacts.models[key]

    if model_key not in artifacts.models:
        raise KeyError(
            f"Model '{model_key}' is not available. Options: {list(artifacts.models.keys())}"
        )
    return model_key, artifacts.models[model_key]


def forecast_horizon(
    artifacts: ForecastArtifacts,
    n_steps: int,
    model_key: Optional[str] = None,
    context_periods: int = config.WEEKLY_LAG + 48,
) -> pd.DataFrame:
    """Recursively forecast ``n_steps`` settlement periods (30 min each)
    beyond the end of the stored history using the chosen model.

    Returns a DataFrame indexed by DATETIME with a single ``forecast_mw``
    column (National Demand, MW).
    """
    used_key, model = _select_model(artifacts, model_key)

    history = artifacts.history.tail(context_periods).copy()
    if config.PRIMARY_TARGET not in history.columns:
        raise ValueError("History is missing the primary target column 'ND'.")

    # Ensure the other two target columns exist for lag-feature purposes;
    # if unavailable, approximate with ND (they move almost in lock-step).
    for col in config.TARGET_COLUMNS:
        if col not in history.columns:
            history[col] = history[config.PRIMARY_TARGET]

    predictions = []
    working_history = history.copy()

    for _ in range(n_steps):
        feature_row = build_feature_row_for_inference(working_history, artifacts.feature_columns)
        X_next = feature_row.to_frame().T
        yhat = float(model.predict(X_next)[0])

        next_time = feature_row.name
        predictions.append((next_time, yhat))

        new_row = {col: yhat for col in config.TARGET_COLUMNS}
        new_row["SETTLEMENT_PERIOD"] = feature_row.get("SETTLEMENT_PERIOD")

        # Carry forward exogenous (non-target) raw columns too, or later
        # rolling/lag windows will slide into NaN-filled predicted rows.
        # Approximate with same-period-yesterday persistence, falling back
        # to the last known observation.
        for col in working_history.columns:
            if col in new_row:
                continue
            series = working_history[col]
            if len(series) >= config.STANDARD_PERIODS_PER_DAY:
                new_row[col] = series.iloc[-config.STANDARD_PERIODS_PER_DAY]
            else:
                new_row[col] = series.iloc[-1]

        working_history.loc[next_time] = pd.Series(new_row)

    result = pd.DataFrame(predictions, columns=["datetime", "forecast_mw"]).set_index("datetime")
    result.attrs["model_used"] = used_key
    return result


def evaluate_on_test_set(artifacts: ForecastArtifacts, model_key: Optional[str] = None) -> dict:
    """Return the stored held-out test metrics for the chosen (or best) model."""
    used_key, _ = _select_model(artifacts, model_key)
    display_name = MODEL_DISPLAY_NAMES.get(used_key, used_key)
    if artifacts.metrics_summary:
        for row in artifacts.metrics_summary:
            if row["Model Name"] == display_name:
                return {"model": display_name, **row}
    return {"model": display_name, "message": "No stored metrics found; run the training pipeline."}

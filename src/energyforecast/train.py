"""End-to-end model training: load the feature matrix, split
chronologically, fit every model, evaluate on the 2025 held-out test set,
and persist both the fitted estimators (``models/*.pkl``) and a metrics
summary (``models/metrics_summary.csv`` / ``.json``).

Converted from ``notebooks/04_model_training_and_evaluation.ipynb``
(sections 1 and 2 — baselines, Linear Regression, Random Forest, XGBoost,
LightGBM and the single-split performance comparison. Hyperparameter
random-search tuning from section 3 is available via
:func:`tune_boosting_models` but is not required for a normal training run).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from . import config, models
from .metrics import ml_error
from .splits import DataSplit, chronological_split, load_feature_matrix, xy

logger = logging.getLogger(__name__)


def train_all_models(
    df: pd.DataFrame,
    models_dir: Path,
    fast_mode: bool = False,
    include_lightgbm: bool = True,
) -> Dict[str, object]:
    """Fit every model on the training split and return the fitted objects."""
    split = chronological_split(df)
    X_train, y_train = xy(split.train, split)
    X_valid, y_valid = xy(split.valid, split)

    fitted: Dict[str, object] = {}
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Linear Regression --------------------------------------------------
    lr = models.linear_regression_model().fit(X_train, y_train)
    fitted["linear_regression"] = lr
    joblib.dump(lr, models_dir / config.MODEL_FILENAMES["linear_regression"])
    logger.info("Trained Linear Regression")

    # --- Random Forest -------------------------------------------------------
    t0 = time.time()
    rf = models.random_forest_model(fast_mode=fast_mode).fit(X_train, y_train)
    fitted["random_forest"] = rf
    joblib.dump(rf, models_dir / config.MODEL_FILENAMES["random_forest"])
    logger.info("Trained Random Forest in %.1fs", time.time() - t0)

    # --- XGBoost --------------------------------------------------------------
    t0 = time.time()
    xgb_model = models.xgboost_model(fast_mode=fast_mode)
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    fitted["xgboost"] = xgb_model
    joblib.dump(xgb_model, models_dir / config.MODEL_FILENAMES["xgboost"])
    logger.info("Trained XGBoost in %.1fs", time.time() - t0)

    # --- LightGBM ---------------------------------------------------------------
    if include_lightgbm:
        import lightgbm as lgb

        t0 = time.time()
        lgbm_model = models.lightgbm_model(fast_mode=fast_mode)
        lgbm_model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        fitted["lightgbm"] = lgbm_model
        joblib.dump(lgbm_model, models_dir / config.MODEL_FILENAMES["lightgbm"])
        logger.info("Trained LightGBM in %.1fs", time.time() - t0)

    return fitted


def evaluate_all_models(df: pd.DataFrame, fitted: Dict[str, object]) -> pd.DataFrame:
    """Evaluate baselines + fitted models on the held-out 2025 test set."""
    split = chronological_split(df)
    X_test, y_test = xy(split.test, split)

    results = []

    snaive_d = models.snaive_daily()
    results.append(ml_error(snaive_d.name, y_test, snaive_d.predict(X_test)))

    snaive_w = models.snaive_weekly()
    results.append(ml_error(snaive_w.name, y_test, snaive_w.predict(X_test)))

    display_names = {
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest Regressor",
        "xgboost": "XGBoost Regressor",
        "lightgbm": "LightGBM Regressor",
    }
    for key, model_obj in fitted.items():
        yhat = model_obj.predict(X_test)
        results.append(ml_error(display_names.get(key, key), y_test, yhat))

    summary = pd.concat(results, ignore_index=True).sort_values("RMSE").reset_index(drop=True)
    return summary


def run_training_pipeline(
    features_csv: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    fast_mode: bool = False,
) -> pd.DataFrame:
    """Full pipeline: load features -> train -> evaluate -> persist artifacts."""
    features_csv = Path(features_csv or config.FEATURES_CSV)
    models_dir = Path(models_dir or config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_matrix(features_csv)
    fitted = train_all_models(df, models_dir, fast_mode=fast_mode)
    summary = evaluate_all_models(df, fitted)

    summary.to_csv(models_dir / "metrics_summary.csv", index=False)
    summary_records = json.loads(summary.to_json(orient="records"))
    (models_dir / "metrics_summary.json").write_text(json.dumps(summary_records, indent=2))

    split = chronological_split(df)
    (models_dir / "feature_columns.json").write_text(json.dumps(split.feature_columns, indent=2))

    logger.info("Training complete. Leaderboard:\n%s", summary.to_string())
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training_pipeline()

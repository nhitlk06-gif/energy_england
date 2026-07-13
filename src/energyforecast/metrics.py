"""Evaluation metrics used throughout the pipeline (from notebook 04)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_percentage_error(y: np.ndarray, yhat: np.ndarray) -> float:
    """Signed mean percentage error — diagnoses systematic over/under bias."""
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.mean((y - yhat) / y))


def mean_absolute_percentage_error(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def ml_error(model_name: str, y: np.ndarray, yhat: np.ndarray) -> pd.DataFrame:
    """R2, MAE, RMSE and MAPE for one set of predictions, as a one-row frame."""
    r2 = r2_score(y, yhat)
    mae = mean_absolute_error(y, yhat)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mape = mean_absolute_percentage_error(y, yhat)
    return pd.DataFrame(
        {"Model Name": model_name, "R2": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape},
        index=[0],
    )


def cross_validation(
    X_training: pd.DataFrame,
    kfold: int,
    model_name: str,
    model,
    feature_cols,
    target_col: str = "ND",
    validation_days: int = 120,
    verbose: bool = False,
) -> pd.DataFrame:
    """Expanding-window time series cross validation.

    Splits ``X_training`` into ``kfold`` consecutive train/validation folds,
    each validation window ``validation_days`` long, walking backwards from
    the most recent data. Mirrors the walk-forward evaluation used in
    notebook 04.
    """
    r2_list, mae_list, rmse_list, mape_list = [], [], [], []

    for k in reversed(range(1, kfold + 1)):
        validation_start = X_training.index.max() - pd.Timedelta(days=k * validation_days)
        validation_end = X_training.index.max() - pd.Timedelta(days=(k - 1) * validation_days)

        training_fold = X_training[X_training.index < validation_start]
        validation_fold = X_training[
            (X_training.index >= validation_start) & (X_training.index <= validation_end)
        ]

        X_train_fold, y_train_fold = training_fold[feature_cols], training_fold[target_col]
        X_valid_fold, y_valid_fold = validation_fold[feature_cols], validation_fold[target_col]

        if verbose:
            print(
                f"[{model_name}] fold k={k}: train={len(X_train_fold):,} rows, "
                f"valid={len(X_valid_fold):,} rows"
            )

        fitted = model.fit(X_train_fold, y_train_fold)
        yhat_fold = fitted.predict(X_valid_fold)
        fold_result = ml_error(model_name, y_valid_fold, yhat_fold)

        r2_list.append(fold_result["R2"][0])
        mae_list.append(fold_result["MAE"][0])
        rmse_list.append(fold_result["RMSE"][0])
        mape_list.append(fold_result["MAPE (%)"][0])

    return pd.DataFrame(
        {
            "Model Name": model_name,
            "R2 CV": f"{np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}",
            "MAE CV": f"{np.mean(mae_list):.2f} +/- {np.std(mae_list):.2f}",
            "RMSE CV": f"{np.mean(rmse_list):.2f} +/- {np.std(rmse_list):.2f}",
            "MAPE CV (%)": f"{np.mean(mape_list):.2f} +/- {np.std(mape_list):.2f}",
        },
        index=[0],
    )

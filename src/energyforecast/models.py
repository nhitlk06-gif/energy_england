"""Model definitions and thin wrappers around the six forecasting models
compared in notebook 04: SNaive Daily, SNaive Weekly, Linear Regression,
Random Forest, XGBoost and LightGBM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from . import config


class Predictor(Protocol):
    def fit(self, X, y): ...
    def predict(self, X): ...


@dataclass
class SNaiveModel:
    """Seasonal-naive baseline: prediction is a single lag column, read
    directly out of the feature matrix. No parameters to fit.
    """

    lag_column: str
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SNaiveModel":
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X[self.lag_column]


def snaive_daily() -> SNaiveModel:
    return SNaiveModel(lag_column=f"{config.PRIMARY_TARGET}_LAG_48", name="SNaive Daily")


def snaive_weekly() -> SNaiveModel:
    return SNaiveModel(lag_column=f"{config.PRIMARY_TARGET}_LAG_336", name="SNaive Weekly")


def linear_regression_model() -> LinearRegression:
    return LinearRegression()


def random_forest_model(fast_mode: bool = False, **overrides) -> RandomForestRegressor:
    params = dict(config.RF_PARAMS)
    if fast_mode:
        params.update(n_estimators=15, max_depth=8)
    params.update(overrides)
    return RandomForestRegressor(**params)


def xgboost_model(fast_mode: bool = False, **overrides):
    import xgboost as xgb

    params = dict(config.XGB_PARAMS)
    if fast_mode:
        params.update(n_estimators=150)
    params.update(overrides)
    return xgb.XGBRegressor(**params)


def lightgbm_model(fast_mode: bool = False, **overrides):
    import lightgbm as lgb

    params = dict(config.LGBM_PARAMS)
    if fast_mode:
        params.update(n_estimators=150)
    params.update(overrides)
    return lgb.LGBMRegressor(**params)

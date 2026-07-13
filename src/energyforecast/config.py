"""Central configuration and constants for the energyforecast pipeline.

All the "magic numbers" that were scattered across the three source
notebooks (01_pre_eda_and_cleaning, 03_feature_engineering_and_selection,
04_model_training_and_evaluation) live here, in one place, so that the
cleaning, feature-engineering, training and inference modules all agree
on the same conventions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
ROOT_DIR = SRC_DIR.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"

CLEANED_CSV = DATA_PROCESSED_DIR / "uk_electricity_cleaned.csv"
FEATURES_CSV = DATA_PROCESSED_DIR / "uk_electricity_features.csv"

# ---------------------------------------------------------------------------
# Raw source files (National Grid ESO historic demand data, 2020-2025)
# ---------------------------------------------------------------------------
RAW_FILE_PATHS = {
    2020: "demanddata_2020.csv",
    2021: "demanddata_2021.csv",
    2022: "demanddata_2022.csv",
    2023: "demanddata_2023.csv",
    2024: "demanddata_2024.csv",
    2025: "demanddata_2025.csv",
}

# Settlement date string formats differ by source year.
SETTLEMENT_DATE_FORMATS = {
    2020: "%d-%b-%Y",
    2021: "%d-%b-%Y",
    2022: "%d-%b-%Y",
    2023: "%d-%b-%y",
    2024: "%d-%b-%Y",
    2025: "%Y-%m-%d",
}

# Column dropped because it only exists from 2023 onward (structural missingness).
STRUCTURALLY_MISSING_COLUMNS = ["SCOTTISH_TRANSFER"]

# Number of standard 30-minute settlement periods per day.
STANDARD_PERIODS_PER_DAY = 48

# Demand columns used for the physical sanity check (<=0 treated as invalid).
DEMAND_COLUMNS = ["ND", "TSD", "ENGLAND_WALES_DEMAND"]

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
TARGET_COLUMNS = ["ND", "TSD", "ENGLAND_WALES_DEMAND"]
PRIMARY_TARGET = "ND"

SHORT_TERM_LAGS = [1, 2, 12, 24, 36, 48]
WEEKLY_LAG = 336

ROLLING_WINDOWS = [8, 24, 48]
SOLAR_COLUMN = "EMBEDDED_SOLAR_GENERATION"

PERIODS_PER_DAY = 48
DAYS_PER_WEEK = 7

MUTUAL_INFO_SAMPLE_SIZE = 15000
MUTUAL_INFO_THRESHOLD = 0.01
MUTUAL_INFO_RANDOM_STATE = 42

# Test set is sealed off during feature selection to avoid leakage.
FEATURE_SELECTION_CUTOFF = "2024-12-31 23:30:00"

ADMINISTRATIVE_COLUMNS_TO_DROP = ["SETTLEMENT_DATE", "DAYOFWEEK", "SOURCE_FILE_YEAR"]

# ---------------------------------------------------------------------------
# Chronological train / valid / test split (matches notebook 04)
# ---------------------------------------------------------------------------
TRAIN_END = "2023-12-31 23:30:00"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31 23:30:00"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31 23:30:00"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model hyperparameters (defaults mirrored from notebook 04)
# ---------------------------------------------------------------------------
RF_PARAMS = dict(n_estimators=50, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE)

XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    early_stopping_rounds=50,
)

LGBM_PARAMS = dict(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
)

MODEL_FILENAMES = {
    "linear_regression": "linear_regression.pkl",
    "random_forest": "random_forest_regressor.pkl",
    "xgboost": "xgboost_regressor.pkl",
    "lightgbm": "lightgbm_regressor.pkl",
}


@dataclass
class PipelineConfig:
    """Bundle of paths/knobs that scripts and the API can override."""

    data_raw_dir: Path = DATA_RAW_DIR
    data_processed_dir: Path = DATA_PROCESSED_DIR
    models_dir: Path = MODELS_DIR
    cleaned_csv: Path = CLEANED_CSV
    features_csv: Path = FEATURES_CSV
    fast_mode: bool = False
    """When True, use lighter hyperparameters so the whole pipeline can be
    exercised quickly (e.g. in CI or a quick local demo)."""

    def ensure_dirs(self) -> None:
        for d in (self.data_processed_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)

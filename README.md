# UK National Grid Electricity Demand Forecasting (England & Wales)

Forecasting **National Demand (ND, MW)** for the Great Britain transmission
system using six years (2020–2025) of half-hourly settlement data published
by **National Grid ESO**. This repository converts the original exploratory
research notebooks into a clean, testable Python package (`src/energyforecast`)
plus a runnable **FastAPI** forecasting service and a **Streamlit** dashboard.

> Converted from `notebooks/01_pre_eda_and_cleaning.ipynb`,
> `notebooks/02_eda.ipynb`, `notebooks/03_feature_engineering_and_selection.ipynb`
> and `notebooks/04_model_training_and_evaluation.ipynb`. The notebooks are
> kept for reference/exploration; the package under `src/` is the
> production-ready, importable, tested version of the same pipeline.

## Results (held-out test year 2025)

| Model | R² | MAE (MW) | RMSE (MW) | MAPE (%) |
|---|---|---|---|---|
| **XGBoost Regressor** | 0.9972 | 247.2 | 325.5 | 0.99 |
| LightGBM Regressor | 0.9970 | 256.8 | 336.7 | 1.03 |
| Random Forest Regressor | 0.9963 | 286.8 | 373.4 | 1.14 |
| Linear Regression | 0.9948 | 339.9 | 444.7 | 1.35 |
| SNaive Daily | 0.8257 | 1849.7 | 2577.2 | 7.37 |
| SNaive Weekly | 0.7764 | 2177.7 | 2918.9 | 8.44 |

XGBoost is the best model on every metric and is used by default by the API
and dashboard.

## Repository layout

```
energy_england/
├── data/
│   ├── raw/                       # 6 source files: demanddata_2020.csv ... demanddata_2025.csv
│   └── processed/                 # generated: uk_electricity_cleaned.csv, uk_electricity_features.csv
├── models/                        # trained *.pkl models + metrics_summary.{csv,json}
├── notebooks/                     # original exploratory notebooks (reference only)
├── src/energyforecast/            # the production package
│   ├── config.py                  # all constants/paths in one place
│   ├── cleaning.py                 # notebook 01 -> raw ingestion, DST resync, interpolation
│   ├── features.py                 # notebook 03 -> lags, rolling stats, cyclical encoding, MI selection
│   ├── splits.py                   # chronological train/valid/test split
│   ├── models.py                   # SNaive / Linear / RandomForest / XGBoost / LightGBM definitions
│   ├── metrics.py                  # R2/MAE/RMSE/MAPE + expanding-window cross validation
│   ├── train.py                    # notebook 04 -> fit + evaluate + persist all models
│   ├── forecast.py                 # inference: recursive multi-step forecasting for the API
│   ├── pipeline.py                 # orchestrates cleaning -> features -> train
│   └── cli.py                      # console-script entry points
├── scripts/                        # thin runnable wrappers around the package
│   ├── run_cleaning.py
│   ├── run_features.py
│   ├── run_training.py
│   └── run_pipeline.py
├── app/
│   ├── api.py                      # FastAPI forecasting service
│   ├── streamlit_app.py            # interactive dashboard
│   └── requirements.txt
├── tests/                          # pytest unit tests for cleaning/features/metrics/API
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

## Quickstart

```bash
git clone <this-repo-url> energy_england
cd energy_england
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -e ".[dev,app,viz]"
```

### 1. Run the full pipeline (clean → features → train)

Trained models are already included under `models/`, so this step is
**optional** unless you want to retrain from scratch or the raw data changes:

```bash
make pipeline          # full hyperparameters (a few minutes on CPU)
make pipeline-fast      # lighter hyperparameters, ~30s, for a quick smoke test
```

Or step by step:

```bash
python scripts/run_cleaning.py     # data/raw/*.csv -> data/processed/uk_electricity_cleaned.csv
python scripts/run_features.py     # -> data/processed/uk_electricity_features.csv
python scripts/run_training.py     # -> models/*.pkl + models/metrics_summary.json
```

### 2. Run the forecasting API

```bash
make api
# or: uvicorn app.api:app --reload --port 8000
```

Open the interactive docs at **http://localhost:8000/docs**. Endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/models` | List trained models with their held-out test metrics |
| GET | `/forecast?steps=48&model=xgboost` | Recursively forecast National Demand `steps` half-hour periods ahead |
| GET | `/history?periods=96` | Most recent observed National Demand |
| GET | `/evaluate?model=xgboost` | Held-out (2025) test metrics for one model |

Example:

```bash
curl "http://localhost:8000/forecast?steps=48" | python -m json.tool
```

### 3. Run the dashboard

```bash
make dashboard
# or: streamlit run app/streamlit_app.py
```

The dashboard talks directly to the trained models on disk by default. To
point it at a separately-running API instance instead:

```bash
ENERGYFORECAST_API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
```

### 4. Run the tests

```bash
make test
# or: pytest
```

## Methodology summary

**Cleaning** (`cleaning.py`, from `01_pre_eda_and_cleaning.ipynb`)
- Merges six yearly CSVs (2020–2025), 105,216 half-hourly rows total.
- Resynchronises the 12 daylight-saving-time transition days (46- or
  50-period days) onto the standard 48-period grid via linear interpolation
  on a normalised within-day time axis — this preserves the diurnal demand
  shape instead of naively dropping/duplicating rows.
- Drops `SCOTTISH_TRANSFER` (structurally missing before 2023).
- Marks non-positive demand readings as missing and linearly interpolates
  all remaining gaps.
- A safety-assertion gate (zero NaNs, zero non-positive demand, exact
  expected row count) must pass before the clean CSV is written.

**Feature engineering** (`features.py`, from
`03_feature_engineering_and_selection.ipynb`)
- Autoregressive lags at `H ∈ {1, 2, 12, 24, 36, 48, 336}` settlement
  periods for all three demand series (`ND`, `TSD`, `ENGLAND_WALES_DEMAND`).
  `ND_LAG_48` and `ND_LAG_336` double as the formulas for the SNaive Daily
  and SNaive Weekly baselines.
- Leakage-safe rolling mean/std (windows 8/24/48 periods), always computed
  on a series shifted by one period first.
- Cyclical sin/cos encoding of settlement-period-of-day (T=48) and
  day-of-week (T=7), plus an `IS_WEEKEND` flag.
- Trims the first 336 rows (insufficient weekly-lag history) and drops
  administrative columns.
- Selects features by **mutual information** against `ND`, computed only on
  the 2020–2024 portion of the data (the 2025 test year is sealed off to
  avoid feature-selection leakage). Threshold: MI > 0.01.
- Final matrix: 104,880 rows × 52 columns (3 targets + 49 selected features).

**Model training** (`train.py`, from
`04_model_training_and_evaluation.ipynb`)
- Chronological split: train 2020–2023, valid 2024 (used for early
  stopping), test 2025 (fully held out, used only once for final scoring).
- Six models compared: SNaive Daily, SNaive Weekly, Linear Regression,
  Random Forest, XGBoost, LightGBM.
- XGBoost/LightGBM use the 2024 validation set for early stopping
  (50 rounds).

**Inference / forecasting** (`forecast.py`)
- Because the models are autoregressive (they consume lag/rolling
  features of the target itself), forecasting beyond the last observed
  settlement period is done **recursively**: each 30-minute-ahead
  prediction is fed back into the history as if it were an observation, so
  the next step's lag/rolling features can be computed, and so on for
  `n_steps`.
- Exogenous raw columns the model also relies on (e.g. embedded
  wind/solar generation) have no future values available at inference
  time; these are approximated with a same-settlement-period-yesterday
  persistence, a standard practical fallback. This makes the forecast a
  pragmatic operational extension of the original historical-evaluation
  notebooks, not a claim that renewable generation is itself being
  forecast.

## Data source

National Grid ESO historic demand data (`demanddata_2020.csv` …
`demanddata_2025.csv`), half-hourly settlement periods, publicly published
by the GB electricity system operator.

## Disclaimer

This is a research/educational forecasting tool. It is **not** an official
National Grid ESO forecast and must not be used for real-time grid
operation, trading, or safety-critical decisions.

## License

MIT — see [LICENSE](LICENSE).

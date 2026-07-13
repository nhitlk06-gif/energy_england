# Architecture

```
demanddata_2020..2025.csv (raw, half-hourly, National Grid ESO)
            │
            ▼
   energyforecast.cleaning
   - merge 6 years, parse dates
   - resync DST days to 48 periods/day
   - drop SCOTTISH_TRANSFER, interpolate gaps/anomalies
   - safety assertions
            │
            ▼
  data/processed/uk_electricity_cleaned.csv  (105,216 rows × 22 cols)
            │
            ▼
   energyforecast.features
   - lag features (H = 1,2,12,24,36,48,336) × {ND, TSD, ENGLAND_WALES_DEMAND}
   - rolling mean/std (windows 8/24/48), leakage-safe (shift(1) first)
   - cyclical sin/cos (period-of-day, day-of-week) + IS_WEEKEND
   - trim first 336 rows, drop admin columns
   - mutual-information feature selection (train+valid only, threshold 0.01)
            │
            ▼
  data/processed/uk_electricity_features.csv (104,880 rows × 52 cols)
            │
            ▼
   energyforecast.splits          (chronological 2020-23 / 2024 / 2025)
            │
            ▼
   energyforecast.train
   - SNaive Daily / Weekly (no fitting — read a lag column)
   - Linear Regression
   - Random Forest
   - XGBoost (early stopping on 2024 valid)
   - LightGBM (early stopping on 2024 valid)
            │
            ▼
  models/*.pkl + models/metrics_summary.{csv,json} + models/feature_columns.json
            │
            ├─────────────────────────────┐
            ▼                             ▼
   app/api.py (FastAPI)          app/streamlit_app.py (dashboard)
   - /forecast, /models,          - talks directly to models/ on disk,
     /history, /evaluate            or to the API via ENERGYFORECAST_API_URL
            │
            ▼
   energyforecast.forecast
   - recursive multi-step forecast beyond the last observed period:
     each prediction is fed back in to build the next step's lag/rolling
     features; exogenous raw columns fall back to same-period-yesterday
     persistence.
```

## Why recursive forecasting?

The models are trained on features derived from the target series itself
(autoregressive lags and rolling statistics). To forecast beyond the last
observed settlement period, there is no "future" value of `ND` to compute
`ND_LAG_1`, `ND_ROLL_MEAN_8`, etc. from — so each predicted value is written
back into the working history as if observed, and the next step's features
are built from that extended history. This is a standard recursive
(iterated) multi-step forecasting strategy. Error can compound over long
horizons, which is why the API caps `steps` at 2000 (about 41 days) and the
dashboard defaults to much shorter horizons.

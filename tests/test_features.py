import numpy as np
import pandas as pd

from energyforecast import features


def _synthetic_series(n=500):
    idx = pd.date_range("2024-01-01", periods=n, freq="30min")
    rng = np.random.default_rng(42)
    base = 25000 + 2000 * np.sin(np.linspace(0, 40 * np.pi, n))
    df = pd.DataFrame(
        {
            "ND": base + rng.normal(0, 50, n),
            "TSD": base * 1.02 + rng.normal(0, 50, n),
            "ENGLAND_WALES_DEMAND": base * 0.9 + rng.normal(0, 50, n),
            "EMBEDDED_SOLAR_GENERATION": np.clip(np.sin(np.linspace(0, 40 * np.pi, n)), 0, None) * 500,
            "SETTLEMENT_PERIOD": (np.arange(n) % 48) + 1,
        },
        index=idx,
    )
    return df


def test_add_lag_features_shapes_and_values():
    df = _synthetic_series()
    out = features.add_lag_features(df, short_term_lags=[1, 2], weekly_lag=48)
    assert "ND_LAG_1" in out.columns
    assert out["ND_LAG_1"].iloc[5] == df["ND"].iloc[4]


def test_add_rolling_features_no_leakage():
    df = _synthetic_series()
    out = features.add_rolling_features(df, windows=[4])
    # rolling mean at position i should only use data up to i-1
    manual = df["ND"].shift(1).rolling(window=4).mean()
    pd.testing.assert_series_equal(out["ND_ROLL_MEAN_4"], manual, check_names=False)


def test_cyclical_features_are_bounded():
    df = _synthetic_series()
    out = features.add_cyclical_calendar_features(df)
    assert out["PERIOD_SIN"].between(-1, 1).all()
    assert out["PERIOD_COS"].between(-1, 1).all()
    assert set(out["IS_WEEKEND"].unique()).issubset({0, 1})


def test_build_feature_row_for_inference_returns_requested_columns():
    df = _synthetic_series(n=400)
    lagged = features.add_lag_features(df, short_term_lags=[1, 2], weekly_lag=48)
    rolled = features.add_rolling_features(lagged, windows=[4, 8])
    calendar = features.add_cyclical_calendar_features(rolled)
    feature_cols = [c for c in calendar.columns if c not in ("ND", "TSD", "ENGLAND_WALES_DEMAND")]

    row = features.build_feature_row_for_inference(df, feature_cols)
    assert list(row.index) == feature_cols
    assert row.name == df.index[-1] + pd.Timedelta(minutes=30)

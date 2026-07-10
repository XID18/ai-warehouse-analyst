import pandas as pd
from ml.forecast import ewm_forecast, build_features


def test_ewm_forecast_returns_float():
    series = pd.Series([10, 12, 9, 15, 14, 16])
    result = ewm_forecast(series)
    assert isinstance(result, float)


def test_build_features_drops_na_rows():
    df = pd.DataFrame({
        "sku": ["A"] * 6,
        "period": pd.date_range("2025-01-01", periods=6, freq="MS"),
        "qty": [10, 12, 9, 15, 14, 16],
    })
    features = build_features(df)
    assert "lag_1" in features.columns
    assert features["lag_1"].isna().sum() == 0

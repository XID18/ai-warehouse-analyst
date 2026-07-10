"""
Прогнозирование спроса по SKU.

Три подхода:
- EWM (экспоненциально взвешенное скользящее среднее) — быстрый baseline
- RandomForestRegressor — учитывает нелинейные паттерны и сезонность
- GradientBoostingRegressor — сравнение точности с RF

Модели персистятся через joblib, чтобы не переобучаться на каждый запрос.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def ewm_forecast(series: pd.Series, alpha: float = 0.3) -> float:
    """Baseline-прогноз: экспоненциально взвешенное среднее."""
    return series.ewm(alpha=alpha).mean().iloc[-1]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ожидает df с колонками ['sku', 'period', 'qty'].
    Строит лаговые и сезонные признаки для одной SKU.
    """
    df = df.sort_values("period").copy()
    df["lag_1"] = df["qty"].shift(1)
    df["lag_2"] = df["qty"].shift(2)
    df["rolling_mean_3"] = df["qty"].rolling(3).mean()
    df["month"] = pd.to_datetime(df["period"]).dt.month
    return df.dropna()


def train_model(df: pd.DataFrame, sku: str, model_type: str = "random_forest"):
    """Обучает модель для одной SKU и сохраняет её на диск."""
    features = build_features(df)
    X = features[["lag_1", "lag_2", "rolling_mean_3", "month"]]
    y = features["qty"]

    model = (
        RandomForestRegressor(n_estimators=200, random_state=42)
        if model_type == "random_forest"
        else GradientBoostingRegressor(random_state=42)
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_DIR / f"{sku}_{model_type}.joblib")
    return model


def predict_next_period(df: pd.DataFrame, sku: str, model_type: str = "random_forest") -> float:
    """
    Прогноз спроса на следующий период по SKU.

    df: история по одной SKU с колонками ['sku', 'period', 'qty']
    Возвращает: прогнозное значение qty на следующий период.
    """
    model_path = MODEL_DIR / f"{sku}_{model_type}.joblib"
    model = joblib.load(model_path) if model_path.exists() else train_model(df, sku, model_type)

    features = build_features(df)
    last_row = features.iloc[[-1]][["lag_1", "lag_2", "rolling_mean_3", "month"]]
    return float(model.predict(last_row)[0])


def evaluate_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)

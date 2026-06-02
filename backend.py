from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_SOURCE = "Yahoo Finance"
DEFAULT_HISTORY_PERIOD = "2y"

TICKER_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
}


def available_tickers() -> list[str]:
    """Возвращает биржевые символы, для которых есть модель и преобразователь."""
    tickers = []
    for metadata_path in sorted(MODEL_DIR.glob("*_metadata.json")):
        ticker = metadata_path.name.replace("_metadata.json", "")
        model_exists = (MODEL_DIR / f"{ticker}_model.pkl").exists()
        scaler_exists = (MODEL_DIR / f"{ticker}_scaler.pkl").exists()
        if model_exists and scaler_exists:
            tickers.append(ticker)
    return tickers


def ticker_label(ticker: str) -> str:
    """Делает понятную подпись для списка акций."""
    company = TICKER_NAMES.get(ticker, ticker)
    return f"{company} ({ticker})"


def extract_close_column(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Достает цену закрытия из таблицы, загруженной через yfinance."""
    if data.empty:
        raise ValueError(f"Не удалось получить данные по биржевому символу {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            close = data[("Close", ticker)]
        else:
            close = data.xs("Close", axis=1, level=0).iloc[:, 0]
    elif "Close" in data.columns:
        close = data["Close"]
    else:
        raise ValueError("В загруженных данных нет столбца с ценой закрытия.")

    close = close.dropna().astype(float)
    if close.empty:
        raise ValueError(f"Для биржевого символа {ticker} нет цен закрытия.")

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = "close"
    return close


def fetch_stock_history(
    ticker: str,
    period: str = DEFAULT_HISTORY_PERIOD,
) -> pd.DataFrame:
    """Загружает историю цен закрытия из внешнего источника."""
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as error:
        raise ValueError(f"Ошибка загрузки данных из {DATA_SOURCE}: {error}") from error

    close = extract_close_column(data, ticker)
    return pd.DataFrame({"close": close})


def regularize_business_days(history: pd.DataFrame) -> pd.Series:
    """Приводит ряд к регулярной шкале рабочих дней."""
    if history.empty or "close" not in history:
        raise ValueError("История цен пуста.")

    series = history["close"].sort_index().asfreq("B").ffill().dropna()
    if len(series) < 40:
        raise ValueError("Слишком мало наблюдений для прогноза.")
    return series


def load_artifacts(ticker: str) -> dict[str, Any]:
    """Загружает сохраненную модель, преобразователь масштаба и сведения о модели."""
    model_path = MODEL_DIR / f"{ticker}_model.pkl"
    scaler_path = MODEL_DIR / f"{ticker}_scaler.pkl"
    metadata_path = MODEL_DIR / f"{ticker}_metadata.json"

    missing = [
        str(path.name)
        for path in (model_path, scaler_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        raise ValueError(
            "Нет обученных файлов: "
            + ", ".join(missing)
            + ". Запустите обучающий блокнот."
        )

    return {
        "model": SARIMAXResults.load(model_path),
        "scaler": joblib.load(scaler_path),
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
    }


def forecast_stock(
    ticker: str,
    horizon: int,
    history_period: str = DEFAULT_HISTORY_PERIOD,
) -> dict[str, Any]:
    """Строит прогноз по готовой модели без повторного обучения."""
    if horizon < 1:
        raise ValueError("Горизонт прогноза должен быть положительным.")

    artifacts = load_artifacts(ticker)
    history = fetch_stock_history(ticker, period=history_period)
    regular_close = regularize_business_days(history)

    scaler = artifacts["scaler"]
    model = artifacts["model"]
    scaled_close = scaler.transform(regular_close.to_numpy().reshape(-1, 1)).ravel()

    try:
        actualized_model = model.apply(scaled_close, refit=False)
    except Exception:
        actualized_model = model

    forecast_scaled = np.asarray(actualized_model.forecast(steps=horizon), dtype=float)
    forecast_prices = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).ravel()
    forecast_prices = np.maximum(forecast_prices, 0)

    last_date = regular_close.index[-1]
    forecast_dates = pd.bdate_range(
        last_date + pd.offsets.BDay(1),
        periods=horizon,
    )
    forecast = pd.DataFrame(
        {
            "date": forecast_dates,
            "forecast_close": forecast_prices,
        }
    )

    latest_close = float(regular_close.iloc[-1])
    final_forecast = float(forecast_prices[-1])

    return {
        "ticker": ticker,
        "history": history,
        "regular_history": pd.DataFrame({"close": regular_close}),
        "forecast": forecast,
        "metadata": artifacts["metadata"],
        "latest_date": last_date,
        "latest_close": latest_close,
        "final_forecast": final_forecast,
        "change_abs": final_forecast - latest_close,
        "change_pct": (final_forecast / latest_close - 1) * 100,
        "source": DATA_SOURCE,
    }

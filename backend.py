from __future__ import annotations  # Отложенное вычисление аннотаций — типы как строки

import json  # Стандартный модуль для работы с JSON
from pathlib import Path  # Объектно-ориентированная работа с путями
from typing import Any  # Универсальный тип для произвольных значений

import joblib  # Сериализация Python-объектов (быстрее pickle для numpy)
import numpy as np  # Библиотека для численных вычислений
import pandas as pd  # Работа с табличными данными и временными рядами
import yfinance as yf  # Загрузка котировок с Yahoo Finance
from statsmodels.tsa.statespace.sarimax import SARIMAXResults  # Класс результатов обученной SARIMAX-модели


BASE_DIR = Path(__file__).resolve().parent  # Абсолютный путь к каталогу текущего файла
MODEL_DIR = BASE_DIR / "models"  # Каталог с обученными моделями и артефактами
DATA_SOURCE = "Yahoo Finance"  # Имя источника данных для отображения в UI
DEFAULT_HISTORY_PERIOD = "2y"  # Период истории по умолчанию — 2 года

TICKER_NAMES = {  # Словарь читаемых названий компаний по тикеру
    "AAPL": "Apple",  # Тикер Apple
    "MSFT": "Microsoft",  # Тикер Microsoft
    "NVDA": "NVIDIA",  # Тикер NVIDIA
}


def available_tickers() -> list[str]:
    """Возвращает биржевые символы, для которых есть модель и преобразователь."""
    tickers = []  # Пустой список под найденные тикеры
    for metadata_path in sorted(MODEL_DIR.glob("*_metadata.json")):  # Перебор всех metadata-файлов
        ticker = metadata_path.name.replace("_metadata.json", "")  # Извлекаем тикер из имени файла
        model_exists = (MODEL_DIR / f"{ticker}_model.pkl").exists()  # Проверяем наличие файла модели
        scaler_exists = (MODEL_DIR / f"{ticker}_scaler.pkl").exists()  # Проверяем наличие файла scaler
        if model_exists and scaler_exists:  # Если оба артефакта на месте
            tickers.append(ticker)  # Добавляем тикер в список
    return tickers  # Возвращаем итоговый список


def ticker_label(ticker: str) -> str:
    """Делает понятную подпись для списка акций."""
    company = TICKER_NAMES.get(ticker, ticker)  # Берём название компании или сам тикер как fallback
    return f"{company} ({ticker})"  # Формат: "Apple (AAPL)"


def extract_close_column(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Достает цену закрытия из таблицы, загруженной через yfinance."""
    if data.empty:  # Проверка на пустой DataFrame
        raise ValueError(f"Не удалось получить данные по биржевому символу {ticker}.")  # Ошибка, если данных нет

    if isinstance(data.columns, pd.MultiIndex):  # Случай мульти-индексных колонок (новые версии yfinance)
        if ("Close", ticker) in data.columns:  # Прямой доступ по паре (поле, тикер)
            close = data[("Close", ticker)]  # Извлекаем нужный столбец
        else:
            close = data.xs("Close", axis=1, level=0).iloc[:, 0]  # Fallback: срез первого уровня
    elif "Close" in data.columns:  # Случай плоских колонок (старые версии yfinance)
        close = data["Close"]  # Прямой доступ к колонке Close
    else:
        raise ValueError("В загруженных данных нет столбца с ценой закрытия.")  # Структура неожиданная

    close = close.dropna().astype(float)  # Удаляем NaN и приводим к float
    if close.empty:  # Если после очистки данных не осталось
        raise ValueError(f"Для биржевого символа {ticker} нет цен закрытия.")  # Бросаем ошибку

    close.index = pd.to_datetime(close.index).tz_localize(None)  # Приводим индекс к datetime без таймзоны
    close.name = "close"  # Имя серии — "close" в нижнем регистре
    return close  # Возвращаем готовую Series


def fetch_stock_history(
    ticker: str,  # Тикер акции
    period: str = DEFAULT_HISTORY_PERIOD,  # Период истории (например "2y")
) -> pd.DataFrame:
    """Загружает историю цен закрытия из внешнего источника."""
    try:
        data = yf.download(  # Запрос к Yahoo Finance
            ticker,  # Какой тикер загрузить
            period=period,  # За какой период
            interval="1d",  # Дневной таймфрейм
            auto_adjust=True,  # Корректировка цен на сплиты и дивиденды
            progress=False,  # Не показывать прогресс-бар
            threads=False,  # Без многопоточности (стабильнее)
        )
    except Exception as error:  # Ловим любую сетевую ошибку
        raise ValueError(f"Ошибка загрузки данных из {DATA_SOURCE}: {error}") from error  # Пробрасываем как ValueError

    close = extract_close_column(data, ticker)  # Извлекаем колонку цен закрытия
    return pd.DataFrame({"close": close})  # Оборачиваем Series в DataFrame с колонкой "close"


def regularize_business_days(history: pd.DataFrame) -> pd.Series:
    """Приводит ряд к регулярной шкале рабочих дней."""
    if history.empty or "close" not in history:  # Проверка корректности входных данных
        raise ValueError("История цен пуста.")  # Ошибка, если входные данные пусты

    series = history["close"].sort_index().asfreq("B").ffill().dropna()  # Сортируем, ресемплируем по business days и заполняем пропуски
    if len(series) < 40:  # Минимум 40 наблюдений для адекватного прогноза
        raise ValueError("Слишком мало наблюдений для прогноза.")  # Иначе ошибка
    return series  # Возвращаем чистый регулярный ряд


def load_artifacts(ticker: str) -> dict[str, Any]:
    """Загружает сохраненную модель, преобразователь масштаба и сведения о модели."""
    model_path = MODEL_DIR / f"{ticker}_model.pkl"  # Путь к файлу модели
    scaler_path = MODEL_DIR / f"{ticker}_scaler.pkl"  # Путь к файлу scaler
    metadata_path = MODEL_DIR / f"{ticker}_metadata.json"  # Путь к JSON с метаданными

    missing = [  # Список отсутствующих файлов
        str(path.name)  # Только имя файла, без полного пути
        for path in (model_path, scaler_path, metadata_path)  # Проверяем все 3 артефакта
        if not path.exists()  # Включаем те, которых нет
    ]
    if missing:  # Если хоть один файл отсутствует
        raise ValueError(  # Бросаем подробную ошибку
            "Нет обученных файлов: "
            + ", ".join(missing)  # Перечисляем имена недостающих файлов
            + ". Запустите обучающий блокнот."
        )

    return {  # Возвращаем словарь с тремя артефактами
        "model": SARIMAXResults.load(model_path),  # Загрузка SARIMAX-модели через statsmodels
        "scaler": joblib.load(scaler_path),  # Десериализация scaler через joblib
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),  # Парсинг JSON метаданных
    }


def forecast_stock(
    ticker: str,  # Тикер акции
    horizon: int,  # Сколько торговых дней прогнозировать
    history_period: str = DEFAULT_HISTORY_PERIOD,  # Период загружаемой истории
) -> dict[str, Any]:
    """Строит прогноз по готовой модели без повторного обучения."""
    if horizon < 1:  # Валидация горизонта
        raise ValueError("Горизонт прогноза должен быть положительным.")  # Ошибка, если ≤ 0

    artifacts = load_artifacts(ticker)  # Загружаем модель + scaler + метаданные
    history = fetch_stock_history(ticker, period=history_period)  # Получаем актуальную историю
    regular_close = regularize_business_days(history)  # Приводим к регулярной business-day шкале

    scaler = artifacts["scaler"]  # Извлекаем scaler из артефактов
    model = artifacts["model"]  # Извлекаем обученную модель
    scaled_close = scaler.transform(regular_close.to_numpy().reshape(-1, 1)).ravel()  # Масштабируем цены и разворачиваем в 1D

    try:
        actualized_model = model.apply(scaled_close, refit=False)  # Применяем модель к новым данным без переобучения
    except Exception:  # На случай несовместимости свежих данных
        actualized_model = model  # Используем исходную модель как fallback

    forecast_scaled = np.asarray(actualized_model.forecast(steps=horizon), dtype=float)  # Прогноз на horizon шагов вперёд
    forecast_prices = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).ravel()  # Обратное преобразование к исходному масштабу
    forecast_prices = np.maximum(forecast_prices, 0)  # Отсекаем отрицательные цены

    last_date = regular_close.index[-1]  # Последняя дата истории
    forecast_dates = pd.bdate_range(  # Генерируем даты прогноза по business-day календарю
        last_date + pd.offsets.BDay(1),  # Старт — следующий рабочий день
        periods=horizon,  # Количество дат = горизонт
    )
    forecast = pd.DataFrame(  # Собираем итоговую таблицу прогноза
        {
            "date": forecast_dates,  # Колонка с датами
            "forecast_close": forecast_prices,  # Колонка с прогнозными ценами
        }
    )

    latest_close = float(regular_close.iloc[-1])  # Последняя известная цена закрытия
    final_forecast = float(forecast_prices[-1])  # Прогнозная цена на конец горизонта

    return {  # Итоговый словарь с результатами для фронтенда
        "ticker": ticker,  # Тикер
        "history": history,  # Сырая история (как пришла)
        "regular_history": pd.DataFrame({"close": regular_close}),  # История на регулярной шкале
        "forecast": forecast,  # Таблица прогноза
        "metadata": artifacts["metadata"],  # Метаданные модели
        "latest_date": last_date,  # Последняя дата истории
        "latest_close": latest_close,  # Последняя цена
        "final_forecast": final_forecast,  # Цена в конце горизонта
        "change_abs": final_forecast - latest_close,  # Абсолютное изменение
        "change_pct": (final_forecast / latest_close - 1) * 100,  # Процентное изменение
        "source": DATA_SOURCE,  # Источник данных
    }

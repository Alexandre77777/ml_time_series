from __future__ import annotations  # Отложенное вычисление аннотаций — типы как строки

import pandas as pd  # Работа с табличными данными
import plotly.graph_objects as go  # Низкоуровневый API Plotly для построения графиков
import streamlit as st  # Фреймворк для создания веб-интерфейсов на Python

from backend import available_tickers, forecast_stock, ticker_label  # Импорт функций из backend-модуля


st.set_page_config(  # Настройка параметров страницы Streamlit
    page_title="Прогнозирование цен акций",  # Заголовок вкладки браузера
    page_icon="chart_with_upwards_trend",  # Иконка вкладки
    layout="wide",  # Широкий макет страницы
)


@st.cache_data(show_spinner=False, ttl=60 * 60)  # Кэш на 1 час, без отображения спиннера
def cached_forecast(ticker: str, horizon: int, history_period: str) -> dict:
    """Сохраняет результат прогноза на один час."""
    return forecast_stock(ticker, horizon, history_period)  # Вызов функции прогноза из backend


def build_forecast_chart(result: dict) -> go.Figure:
    """Создает график фактических цен и будущего прогноза."""
    history = result["regular_history"].tail(180)  # Берём последние 180 дней истории
    forecast = result["forecast"]  # Извлекаем данные прогноза

    fig = go.Figure()  # Создаём пустую фигуру Plotly
    fig.add_trace(  # Добавляем линию исторических цен
        go.Scatter(
            x=history.index,  # По оси X — даты (индекс DataFrame)
            y=history["close"],  # По оси Y — цены закрытия
            mode="lines",  # Тип отображения: линия
            name="Фактическая цена",  # Подпись в легенде
            line=dict(color="#2563eb", width=2),  # Синяя линия толщиной 2
        )
    )
    fig.add_trace(  # Добавляем линию прогнозных цен
        go.Scatter(
            x=forecast["date"],  # Даты прогноза по X
            y=forecast["forecast_close"],  # Прогнозные цены по Y
            mode="lines+markers",  # Линия с маркерами на точках
            name="Прогноз",  # Подпись в легенде
            line=dict(color="#dc2626", width=2),  # Красная линия толщиной 2
            marker=dict(size=5),  # Размер маркеров
        )
    )
    fig.add_vline(  # Вертикальная линия — граница между историей и прогнозом
        x=result["latest_date"],  # Координата X — последняя известная дата
        line_width=1,  # Толщина линии
        line_dash="dash",  # Пунктирный стиль
        line_color="#6b7280",  # Серый цвет
    )
    fig.update_layout(  # Настройка общего внешнего вида графика
        height=520,  # Высота графика в пикселях
        margin=dict(l=20, r=20, t=30, b=20),  # Внешние отступы
        hovermode="x unified",  # Единый тултип по оси X при наведении
        legend=dict(orientation="h", y=1.04, x=0),  # Горизонтальная легенда сверху
        xaxis_title="Дата",  # Подпись оси X
        yaxis_title="Цена закрытия, доллары США",  # Подпись оси Y
    )
    return fig  # Возвращаем готовую фигуру


def show_model_summary(metadata: dict) -> None:
    """Показывает параметры лучшей модели из обучающего блокнота."""
    order = tuple(metadata.get("order", []))  # Параметры (p, d, q) модели
    seasonal_order = tuple(metadata.get("seasonal_order", []))  # Сезонные параметры (P, D, Q, s)
    metrics = metadata.get("metrics", {})  # Словарь метрик качества
    selection_method = metadata.get("selection_method", "pmdarima.auto_arima")  # Метод подбора по умолчанию
    selection_criterion = metadata.get(  # Критерий выбора с fallback
        "selection_criterion",
        "метрики на контрольном отрезке",
    )

    st.subheader("Итоги обучения")  # Подзаголовок секции
    col1, col2, col3, col4 = st.columns(4)  # Создаём 4 колонки одинаковой ширины
    col1.metric("Семейство", metadata.get("family", "-"))  # Семейство моделей (ARIMA/SARIMAX и т.п.)
    col2.metric("Параметры", str(order))  # Отображение параметров order
    col3.metric("Сезонные параметры", str(seasonal_order))  # Отображение сезонных параметров
    col4.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")  # Метрика MAPE с двумя знаками после запятой
    st.caption(f"Метод подбора: {selection_method}; критерий: {selection_criterion}.")  # Поясняющая подпись

    with st.expander("Метрики лучшей модели"):  # Раскрывающийся блок с метриками
        metric_table = pd.DataFrame(  # Создаём таблицу с метриками одной строкой
            [
                {
                    "MAE": metrics.get("mae"),  # Средняя абсолютная ошибка
                    "RMSE": metrics.get("rmse"),  # Корень из среднеквадратичной ошибки
                    "MAPE, %": metrics.get("mape"),  # Средняя абсолютная ошибка в процентах
                    "AIC": metadata.get("aic"),  # Информационный критерий Акаике
                    "BIC": metadata.get("bic"),  # Байесовский информационный критерий
                }
            ]
        )
        st.dataframe(metric_table, use_container_width=True)  # Выводим DataFrame на всю ширину


def main() -> None:
    st.title("Прогнозирование цен акций")  # Главный заголовок приложения
    st.caption(  # Описание приложения мелким шрифтом
        "Учебное приложение Streamlit. Пользовательская часть и модуль "
        "загрузки данных вынесены в отдельные файлы."
    )

    tickers = available_tickers()  # Получаем список доступных тикеров
    if not tickers:  # Если моделей нет — выводим ошибку
        st.error(
            "Модели не найдены. Сначала выполните обучающий блокнот "
            "notebooks/training_pipeline_auto_arima_stock_forecasting.ipynb."
        )
        return  # Прекращаем выполнение функции

    with st.sidebar:  # Контекст боковой панели
        st.header("Параметры прогноза")  # Заголовок боковой панели
        ticker = st.selectbox(  # Выпадающий список тикеров
            "Акция",  # Подпись
            tickers,  # Варианты выбора
            format_func=ticker_label,  # Функция форматирования отображения
        )
        horizon = st.slider(  # Слайдер для выбора горизонта прогноза
            "Горизонт прогноза, торговых дней",  # Подпись
            min_value=1,  # Минимум — 1 день
            max_value=60,  # Максимум — 60 дней
            value=14,  # Значение по умолчанию — 14 дней
        )
        history_period = st.selectbox(  # Выбор исторического периода
            "История для актуализации",  # Подпись
            ["6mo", "1y", "2y", "5y"],  # Варианты периодов
            index=2,  # По умолчанию выбран "2y"
        )
        run_forecast = st.button("Построить прогноз", type="primary")  # Главная кнопка запуска

        st.divider()  # Горизонтальный разделитель
        st.info(  # Информационный блок с дисклеймером
            "Данные загружаются из Yahoo Finance. Прогноз является учебным "
            "примером и не является финансовой рекомендацией."
        )

    current_params = (ticker, horizon, history_period)  # Текущий набор параметров
    cached_params = st.session_state.get("last_params")  # Параметры из прошлого запуска

    if (  # Условие переиспользования закэшированного результата
        not run_forecast  # Кнопка не нажата
        and "last_result" in st.session_state  # Прошлый результат сохранён
        and cached_params == current_params  # Параметры не изменились
    ):
        result = st.session_state.last_result  # Берём результат из session_state
    else:
        with st.spinner("Загружаю цены и строю прогноз..."):  # Показ спиннера на время вычислений
            try:
                result = cached_forecast(ticker, horizon, history_period)  # Запрос прогноза с кэшем
                st.session_state.last_result = result  # Сохраняем результат
                st.session_state.last_params = current_params  # Сохраняем параметры
            except ValueError as error:  # Обработка ошибок валидации
                st.error(str(error))  # Показываем сообщение об ошибке
                return  # Прерываем выполнение

    col1, col2, col3, col4 = st.columns(4)  # 4 колонки для ключевых метрик
    col1.metric("Биржевой символ", result["ticker"])  # Отображение тикера
    col2.metric(  # Последняя известная цена
        "Последняя цена",
        f"${result['latest_close']:.2f}",  # Цена с 2 знаками после запятой
        help=f"Дата: {result['latest_date'].date()}",  # Подсказка с датой
    )
    col3.metric("Прогноз в конце горизонта", f"${result['final_forecast']:.2f}")  # Прогнозная цена в конце
    col4.metric(  # Изменение цены — абсолютное и в процентах
        "Изменение",
        f"{result['change_abs']:+.2f}",  # Абсолютное изменение со знаком
        f"{result['change_pct']:+.2f}%",  # Процентное изменение со знаком
    )

    st.plotly_chart(build_forecast_chart(result), use_container_width=True)  # Отображаем график на всю ширину

    forecast_table = result["forecast"].copy()  # Копия таблицы прогноза для форматирования
    forecast_table["date"] = forecast_table["date"].dt.strftime("%Y-%m-%d")  # Преобразуем даты в строки
    forecast_table["forecast_close"] = forecast_table["forecast_close"].round(2)  # Округляем прогнозы

    left, right = st.columns([2, 1])  # Две колонки в пропорции 2:1
    with left:  # Левая колонка
        st.subheader("Таблица прогноза")  # Подзаголовок
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)  # Таблица без индекса
        st.download_button(  # Кнопка скачивания CSV
            "Скачать прогноз в CSV",  # Текст кнопки
            forecast_table.to_csv(index=False).encode("utf-8"),  # Содержимое в байтах UTF-8
            file_name=f"{result['ticker']}_forecast.csv",  # Имя скачиваемого файла
            mime="text/csv",  # MIME-тип файла
        )

    with right:  # Правая колонка
        show_model_summary(result["metadata"])  # Сводка по модели


if __name__ == "__main__":  # Точка входа: запуск только при прямом исполнении файла
    main()  # Вызов главной функции

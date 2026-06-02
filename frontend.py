from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend import available_tickers, forecast_stock, ticker_label


st.set_page_config(
    page_title="Прогнозирование цен акций",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_forecast(ticker: str, horizon: int, history_period: str) -> dict:
    """Сохраняет результат прогноза на один час."""
    return forecast_stock(ticker, horizon, history_period)


def build_forecast_chart(result: dict) -> go.Figure:
    """Создает график фактических цен и будущего прогноза."""
    history = result["regular_history"].tail(180)
    forecast = result["forecast"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["close"],
            mode="lines",
            name="Фактическая цена",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["forecast_close"],
            mode="lines+markers",
            name="Прогноз",
            line=dict(color="#dc2626", width=2),
            marker=dict(size=5),
        )
    )
    fig.add_vline(
        x=result["latest_date"],
        line_width=1,
        line_dash="dash",
        line_color="#6b7280",
    )
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0),
        xaxis_title="Дата",
        yaxis_title="Цена закрытия, доллары США",
    )
    return fig


def show_model_summary(metadata: dict) -> None:
    """Показывает параметры лучшей модели из обучающего блокнота."""
    order = tuple(metadata.get("order", []))
    seasonal_order = tuple(metadata.get("seasonal_order", []))
    metrics = metadata.get("metrics", {})
    selection_method = metadata.get("selection_method", "ручной подбор параметров")
    selection_criterion = metadata.get(
        "selection_criterion",
        "метрики на контрольном отрезке",
    )

    st.subheader("Итоги обучения")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Семейство", metadata.get("family", "-"))
    col2.metric("Параметры", str(order))
    col3.metric("Сезонные параметры", str(seasonal_order))
    col4.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
    st.caption(f"Метод подбора: {selection_method}; критерий: {selection_criterion}.")

    with st.expander("Метрики лучшей модели"):
        metric_table = pd.DataFrame(
            [
                {
                    "MAE": metrics.get("mae"),
                    "RMSE": metrics.get("rmse"),
                    "MAPE, %": metrics.get("mape"),
                    "AIC": metadata.get("aic"),
                    "BIC": metadata.get("bic"),
                }
            ]
        )
        st.dataframe(metric_table, use_container_width=True)


def main() -> None:
    st.title("Прогнозирование цен акций")
    st.caption(
        "Учебное приложение Streamlit. Пользовательская часть и модуль "
        "загрузки данных вынесены в отдельные файлы."
    )

    tickers = available_tickers()
    if not tickers:
        st.error(
            "Модели не найдены. Сначала выполните обучающий блокнот "
            "notebooks/training_pipeline_auto_arima_stock_forecasting.ipynb."
        )
        return

    with st.sidebar:
        st.header("Параметры прогноза")
        ticker = st.selectbox(
            "Акция",
            tickers,
            format_func=ticker_label,
        )
        horizon = st.slider(
            "Горизонт прогноза, торговых дней",
            min_value=1,
            max_value=60,
            value=14,
        )
        history_period = st.selectbox(
            "История для актуализации",
            ["6mo", "1y", "2y", "5y"],
            index=2,
        )
        run_forecast = st.button("Построить прогноз", type="primary")

        st.divider()
        st.info(
            "Данные загружаются из Yahoo Finance. Прогноз является учебным "
            "примером и не является финансовой рекомендацией."
        )

    current_params = (ticker, horizon, history_period)
    cached_params = st.session_state.get("last_params")

    if (
        not run_forecast
        and "last_result" in st.session_state
        and cached_params == current_params
    ):
        result = st.session_state.last_result
    else:
        with st.spinner("Загружаю цены и строю прогноз..."):
            try:
                result = cached_forecast(ticker, horizon, history_period)
                st.session_state.last_result = result
                st.session_state.last_params = current_params
            except ValueError as error:
                st.error(str(error))
                return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Биржевой символ", result["ticker"])
    col2.metric(
        "Последняя цена",
        f"${result['latest_close']:.2f}",
        help=f"Дата: {result['latest_date'].date()}",
    )
    col3.metric("Прогноз в конце горизонта", f"${result['final_forecast']:.2f}")
    col4.metric(
        "Изменение",
        f"{result['change_abs']:+.2f}",
        f"{result['change_pct']:+.2f}%",
    )

    st.plotly_chart(build_forecast_chart(result), use_container_width=True)

    forecast_table = result["forecast"].copy()
    forecast_table["date"] = forecast_table["date"].dt.strftime("%Y-%m-%d")
    forecast_table["forecast_close"] = forecast_table["forecast_close"].round(2)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Таблица прогноза")
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        st.download_button(
            "Скачать прогноз в CSV",
            forecast_table.to_csv(index=False).encode("utf-8"),
            file_name=f"{result['ticker']}_forecast.csv",
            mime="text/csv",
        )

    with right:
        show_model_summary(result["metadata"])


if __name__ == "__main__":
    main()

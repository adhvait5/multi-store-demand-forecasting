"""
Exploratory data analysis utilities for time series.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def trend_decomposition(
    series: pd.Series,
    period: int = 7,
    model: str = "additive",
) -> Tuple[pd.DataFrame, object]:
    """
    Decompose series into trend, seasonal, residual.
    Returns (components DataFrame, decomposition object).
    """
    decomp = seasonal_decompose(series.dropna(), model=model, period=period, extrapolate_trend="freq")
    components = pd.DataFrame(
        {
            "trend": decomp.trend,
            "seasonal": decomp.seasonal,
            "residual": decomp.resid,
        }
    )
    return components, decomp


def adf_stationarity_test(series: pd.Series) -> dict:
    """Run Augmented Dickey-Fuller test. Returns dict with statistic, p-value, interpretation."""
    series_clean = series.dropna()
    result = adfuller(series_clean, autolag="AIC")
    stat, pval = result[0], result[1]
    is_stationary = pval < 0.05
    return {
        "adf_statistic": stat,
        "p_value": pval,
        "is_stationary": is_stationary,
        "interpretation": "Stationary (reject H0)" if is_stationary else "Non-stationary (fail to reject H0)",
    }


def plot_acf_pacf(series: pd.Series, lags: int = 40, figsize: Tuple[int, int] = (12, 5)):
    """Plot ACF and PACF for ARIMA order selection."""
    series_clean = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_acf(series_clean, lags=lags, ax=axes[0])
    plot_pacf(series_clean, lags=lags, ax=axes[1], method="ywm")
    plt.tight_layout()
    return fig


def analyze_holiday_effects(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Compare mean sales on holiday vs non-holiday dates."""
    if "is_holiday" not in df.columns:
        return pd.DataFrame()
    daily = df.groupby(date_col).agg({"sales": "sum", "is_holiday": "first"}).reset_index()
    return daily.groupby("is_holiday")["sales"].agg(["mean", "std", "count"])


def analyze_seasonality(
    df: pd.DataFrame,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze weekly and monthly seasonality. Returns (weekly means, monthly means)."""
    df = df.copy()
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    daily = df.groupby(date_col).agg({"sales": "sum", "day_of_week": "first", "month": "first"})
    weekly = daily.groupby("day_of_week")["sales"].mean()
    monthly = daily.groupby("month")["sales"].mean()
    return weekly, monthly


def plot_decomposition(decomp_obj, figsize: Tuple[int, int] = (12, 8)):
    """Plot seasonal decomposition (trend, seasonal, residual)."""
    return decomp_obj.plot()

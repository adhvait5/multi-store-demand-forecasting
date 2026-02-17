"""
Statistical time series models: ARIMA, SARIMA, SARIMAX.
"""
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats


def fit_sarima(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    *,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> tuple:
    """Fit SARIMA model. Returns (model, fitted_result).
    If default fit fails, retries with enforce_stationarity=False, enforce_invertibility=False.
    """
    so = seasonal_order if seasonal_order != (0, 0, 0, 0) else (0, 0, 0, 0)
    try:
        model = SARIMAX(
            series.dropna(),
            order=order,
            seasonal_order=so,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        result = model.fit(disp=False)
        return model, result
    except Exception:
        model = SARIMAX(
            series.dropna(),
            order=order,
            seasonal_order=so,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
        return model, result


def fit_sarimax(
    series: pd.Series,
    exog: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple:
    """Fit SARIMAX with exogenous variables. Returns (model, fitted_result)."""
    model = SARIMAX(
        series.dropna(),
        exog=exog.loc[series.dropna().index].fillna(0),
        order=order,
        seasonal_order=seasonal_order if seasonal_order != (0, 0, 0, 0) else (0, 0, 0, 0),
    )
    result = model.fit(disp=False)
    return model, result


def _fit_sarima_with_fallback(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    metric: str = "aic",
) -> tuple[Optional[object], Optional[float]]:
    """Try fit with default, then relaxed stationarity/invertibility. Returns (result, score) or (None, None)."""
    for enforce_stationarity, enforce_invertibility in [(True, True), (False, False)]:
        try:
            _, res = fit_sarima(
                series, order, seasonal_order,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            )
            score = getattr(res, metric.lower())
            return res, score
        except Exception:
            continue
    return None, None


def grid_search_sarima(
    series: pd.Series,
    p_range: tuple = (0, 2),
    d_range: tuple = (0, 1),
    q_range: tuple = (0, 2),
    P_range: tuple = (0, 1),
    D_range: tuple = (0, 1),
    Q_range: tuple = (0, 1),
    s: int = 7,
    metric: str = "aic",
) -> dict:
    """
    Grid search over SARIMA orders. Returns best params and metric.
    If all combinations fail, tries fallback orders with relaxed constraints.
    """
    best_score = np.inf
    best_params = None
    results = []

    for p, d, q in product(range(p_range[0], p_range[1] + 1), range(d_range[0], d_range[1] + 1), range(q_range[0], q_range[1] + 1)):
        for P, D, Q in product(range(P_range[0], P_range[1] + 1), range(D_range[0], D_range[1] + 1), range(Q_range[0], Q_range[1] + 1)):
            if p == 0 and q == 0 and P == 0 and Q == 0:
                continue
            res, score = _fit_sarima_with_fallback(series, (p, d, q), (P, D, Q, s), metric)
            if res is not None:
                results.append({"order": (p, d, q), "seasonal_order": (P, D, Q, s), metric: score})
                if score < best_score:
                    best_score = score
                    best_params = {"order": (p, d, q), "seasonal_order": (P, D, Q, s)}

    # Fallback: try common robust orders with relaxed constraints when grid fails
    if best_params is None:
        fallbacks = [
            ((1, 0, 1), (0, 1, 1, s)),  # SARIMA(1,0,1)(0,1,1,7) - common for daily data
            ((0, 1, 1), (0, 1, 1, s)),  # SARIMA(0,1,1)(0,1,1,7)
            ((1, 1, 0), (0, 1, 1, s)),  # SARIMA(1,1,0)(0,1,1,7)
            ((0, 1, 1), (0, 0, 0, s)),  # ARIMA(0,1,1) - no seasonal
            ((1, 0, 0), (0, 1, 0, s)),  # SARIMA(1,0,0)(0,1,0,7)
        ]
        for order, seasonal_order in fallbacks:
            res, score = _fit_sarima_with_fallback(series, order, seasonal_order, metric)
            if res is not None:
                best_score = score
                best_params = {"order": order, "seasonal_order": seasonal_order}
                results.append({"order": order, "seasonal_order": seasonal_order, metric: score})
                break

    return {"best_params": best_params, "best_score": best_score, "all_results": results}


def residual_diagnostics(result, figsize=(12, 8)):
    """Plot residuals, ACF of residuals, and QQ-plot."""
    import matplotlib.pyplot as plt

    residuals = result.resid.dropna()
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    axes[0].plot(residuals)
    axes[0].set_title("Residuals over time")
    axes[0].set_ylabel("Residual")

    plot_acf(residuals, lags=40, ax=axes[1])

    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q plot")

    plt.tight_layout()
    return fig


def forecast_sarima(result, steps: int, exog_future: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Forecast next steps. For SARIMAX, pass exog_future."""
    f = result.forecast(steps=steps, exog=exog_future)
    return np.array(f)

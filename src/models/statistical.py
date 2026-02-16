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
) -> tuple:
    """Fit SARIMA model. Returns (model, fitted_result)."""
    model = SARIMAX(
        series.dropna(),
        order=order,
        seasonal_order=seasonal_order if seasonal_order != (0, 0, 0, 0) else (0, 0, 0, 0),
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
    """
    best_score = np.inf
    best_params = None
    results = []

    for p, d, q in product(range(p_range[0], p_range[1] + 1), range(d_range[0], d_range[1] + 1), range(q_range[0], q_range[1] + 1)):
        for P, D, Q in product(range(P_range[0], P_range[1] + 1), range(D_range[0], D_range[1] + 1), range(Q_range[0], Q_range[1] + 1)):
            if p == 0 and q == 0 and P == 0 and Q == 0:
                continue
            try:
                _, res = fit_sarima(series, (p, d, q), (P, D, Q, s))
                score = getattr(res, metric.upper())
                results.append({"order": (p, d, q), "seasonal_order": (P, D, Q, s), metric: score})
                if score < best_score:
                    best_score = score
                    best_params = {"order": (p, d, q), "seasonal_order": (P, D, Q, s)}
            except Exception:
                continue

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

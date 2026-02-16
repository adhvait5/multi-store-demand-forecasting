"""
Baseline forecasting models for benchmarking.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class NaiveForecast:
    """Predict last observed value."""

    def fit(self, y: np.ndarray):
        self.last_ = y[-1] if len(y) > 0 else 0.0
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last_)


class MovingAverageForecast:
    """Predict rolling mean of last `window` observations."""

    def __init__(self, window: int = 7):
        self.window = window

    def fit(self, y: np.ndarray):
        y = np.array(y)
        self.mean_ = np.mean(y[-self.window :]) if len(y) >= self.window else np.mean(y)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.mean_)


class LinearRegressionWithLags:
    """Linear regression on lag features (t-1, t-7, t-14, t-30)."""

    def __init__(self, lags: list = None):
        self.lags = lags or [1, 7, 14, 30]
        self.model_ = LinearRegression()
        self.scaler_ = StandardScaler()
        self.last_values_ = None

    def _build_features(self, series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.DataFrame(index=series.index)
        for lag in sorted(self.lags):
            df[f"lag_{lag}"] = series.shift(lag)
        df = df.dropna()
        X = df
        y = series.loc[df.index]
        return X, y

    def fit(self, series: pd.Series):
        X, y = self._build_features(series)
        X_scaled = self.scaler_.fit_transform(X)
        self.model_.fit(X_scaled, y)
        # Keep enough history for max lag
        self.history_ = list(series.iloc[-max(self.lags) :].values)
        return self

    def predict(self, horizon: int, series: pd.Series = None) -> np.ndarray:
        """Recursive prediction for next `horizon` steps."""
        vals = list(series.iloc[-max(self.lags) :].values) if series is not None else list(self.history_)
        preds = []
        for _ in range(horizon):
            X = np.array([[vals[-lag] if len(vals) >= lag else vals[0] for lag in sorted(self.lags)]])
            X_scaled = self.scaler_.transform(X)
            p = self.model_.predict(X_scaled)[0]
            preds.append(p)
            vals.append(p)
        return np.array(preds)

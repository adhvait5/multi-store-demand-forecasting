"""
Machine learning forecasting models with time-series cross-validation.
"""
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb


def get_time_series_cv(n_splits: int = 5):
    """Time-series cross-validation with expanding window."""
    return TimeSeriesSplit(n_splits=n_splits, test_size=None)


def fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    **kwargs,
) -> xgb.XGBRegressor:
    """Train XGBoost with optional early stopping."""
    params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "random_state": 42, **kwargs}
    model = xgb.XGBRegressor(**params)
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)
    return model


def fit_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    **kwargs,
) -> lgb.LGBMRegressor:
    """Train LightGBM with optional early stopping."""
    params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "random_state": 42, "verbose": -1, **kwargs}
    model = lgb.LGBMRegressor(**params)
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def fit_random_forest(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestRegressor:
    """Train Random Forest."""
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42, **kwargs}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    models: dict[str, Any],
    cv: TimeSeriesSplit,
    metrics_fn,
) -> dict[str, list]:
    """Run time-series CV and return metric per fold per model."""
    results = {name: [] for name in models}
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        for name, model_cls in models.items():
            if "XGB" in name.upper() or "LightGBM" in name.upper():
                m = model_cls(X_train, y_train, X_val, y_val)
            else:
                m = model_cls(X_train, y_train)
            pred = m.predict(X_val)
            score = metrics_fn(y_val, pred)
            results[name].append(score)
    return results

"""
Evaluation metrics for forecasting models.
"""
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error. Clips zeros to avoid inf."""
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    y_true = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_forecasts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    store_ids: np.ndarray = None,
) -> dict:
    """
    Compute MAE, RMSE, MAPE. Optionally compute weighted error by store.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    results = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
    if store_ids is not None and len(store_ids) == len(y_true):
        # Weighted MAE by store (equal weight per store, then average)
        store_ids = np.array(store_ids)
        stores = np.unique(store_ids)
        store_maes = []
        for s in stores:
            mask = store_ids == s
            if mask.sum() > 0:
                store_maes.append(mae(y_true[mask], y_pred[mask]))
        if store_maes:
            results["MAE_per_store_mean"] = np.mean(store_maes)
    return results

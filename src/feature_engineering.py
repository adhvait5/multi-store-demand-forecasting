"""
Feature engineering for ML forecasting models.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"


def build_lag_features(
    df: pd.DataFrame,
    value_col: str = "sales",
    date_col: str = "date",
    group_cols: Optional[list] = None,
    lags: list = None,
) -> pd.DataFrame:
    """Add lag features (t-1, t-7, t-14, t-30). df should be sorted by date (and group_cols if used)."""
    lags = lags or [1, 7, 14, 30]
    out = df.copy()
    if group_cols:
        for lag in lags:
            out[f"lag_{lag}"] = out.groupby(group_cols)[value_col].shift(lag)
    else:
        out = out.sort_values(date_col).reset_index(drop=True)
        for lag in lags:
            out[f"lag_{lag}"] = out[value_col].shift(lag)
    return out


def build_rolling_features(
    df: pd.DataFrame,
    value_col: str = "sales",
    date_col: str = "date",
    group_cols: Optional[list] = None,
    windows: list = None,
) -> pd.DataFrame:
    """Add rolling mean features (shifted by 1 to avoid leakage)."""
    windows = windows or [7, 14, 30]
    out = df.copy()
    if group_cols:
        g = out.groupby(group_cols)[value_col]
        for w in windows:
            out[f"rolling_mean_{w}"] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    else:
        out = out.sort_values(date_col).reset_index(drop=True)
        for w in windows:
            out[f"rolling_mean_{w}"] = out[value_col].shift(1).rolling(w, min_periods=1).mean()
    return out


def add_holiday_indicator(df: pd.DataFrame, holidays_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Add is_holiday column. Uses existing if in df, else from holidays_df."""
    if "is_holiday" in df.columns:
        return df
    if holidays_df is not None and "date" in holidays_df.columns:
        holiday_dates = holidays_df[holidays_df["locale"] == "National"]["date"].unique()
        df = df.copy()
        df["is_holiday"] = df["date"].isin(holiday_dates)
    return df


def add_store_encoding(df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    """Add store-level categorical encodings (city, state, type, cluster)."""
    out = df.merge(stores_df, on="store_nbr", how="left")
    for col in ["city", "state", "type"]:
        if col in out.columns:
            out[f"{col}_code"] = pd.Categorical(out[col]).codes
    return out


def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add day_of_week, month, year, is_weekend."""
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["year"] = dt.dt.year
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return out


def build_ml_features(
    df: pd.DataFrame,
    stores_df: Optional[pd.DataFrame] = None,
    holidays_df: Optional[pd.DataFrame] = None,
    lags: list = None,
    rolling_windows: list = None,
    group_cols: Optional[list] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Build full feature matrix for ML. Expects df with date, sales, and optionally onpromotion, is_holiday.
    For daily aggregate: single row per date. For store-family level: use group_cols=['store_nbr','family'].
    """
    out = df.copy()
    out = add_temporal_features(out, date_col=date_col)
    if "onpromotion" in out.columns:
        out["promotion_flag"] = (out["onpromotion"] > 0).astype(int)
    if stores_df is not None and "store_nbr" in out.columns:
        out = add_store_encoding(out, stores_df)
    if holidays_df is not None or "is_holiday" in out.columns:
        out = add_holiday_indicator(out, holidays_df)
    out = build_lag_features(out, date_col=date_col, group_cols=group_cols, lags=lags or [1, 7, 14, 30])
    out = build_rolling_features(out, date_col=date_col, group_cols=group_cols, windows=rolling_windows or [7, 14, 30])
    return out

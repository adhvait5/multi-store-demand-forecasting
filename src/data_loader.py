"""
Data loading and merging for the Store Sales - Time Series Forecasting dataset.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"


def load_raw_tables(data_dir: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    """Load all raw CSV files into a dictionary of DataFrames."""
    data_dir = data_dir or DEFAULT_RAW_DIR
    tables = {}
    files = [
        "train.csv",
        "test.csv",
        "stores.csv",
        "oil.csv",
        "holidays_events.csv",
        "transactions.csv",
    ]
    for f in files:
        path = data_dir / f
        if path.exists():
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            tables[f.replace(".csv", "")] = df
    return tables


def load_and_merge_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load train data and merge with stores, oil, holidays, and transactions.
    Returns a single DataFrame for analysis.
    """
    tables = load_raw_tables(data_dir)
    train = tables.get("train")
    if train is None:
        raise FileNotFoundError(
            f"train.csv not found in {data_dir or DEFAULT_RAW_DIR}. "
            "Run scripts/download_data.py first."
        )

    # Merge stores
    if "stores" in tables:
        train = train.merge(tables["stores"], on="store_nbr", how="left")

    # Merge oil (forward fill for missing dates)
    if "oil" in tables:
        oil = tables["oil"].rename(columns={"dcoilwtico": "oil_price"})
        train = train.merge(oil, on="date", how="left")
        train["oil_price"] = train["oil_price"].ffill()

    # Merge holidays - mark dates that have any holiday/event
    if "holidays_events" in tables:
        holidays = tables["holidays_events"]
        # National holidays or transferred holidays (actual day off)
        holiday_dates = holidays[
            (holidays["locale"] == "National") | (holidays["transferred"])
        ]["date"].unique()
        train["is_holiday"] = train["date"].isin(holiday_dates)

    # Merge transactions (store-day level)
    if "transactions" in tables:
        train = train.merge(
            tables["transactions"],
            on=["date", "store_nbr"],
            how="left",
        )

    return train


def get_aggregated_series(df: pd.DataFrame) -> pd.Series:
    """Aggregate total sales by date."""
    agg = df.groupby("date")["sales"].sum()
    agg.index = pd.to_datetime(agg.index)
    return agg.sort_index()


def get_store_family_series(
    df: pd.DataFrame,
    store_nbr: int,
    family: str,
) -> pd.Series:
    """Get sales series for a specific store and product family."""
    subset = df[(df["store_nbr"] == store_nbr) & (df["family"] == family)]
    return get_aggregated_series(subset)

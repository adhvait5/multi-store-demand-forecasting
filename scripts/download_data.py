"""
Download the Store Sales - Time Series Forecasting dataset from Kaggle.
Requires Kaggle API credentials at ~/.kaggle/kaggle.json
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
COMPETITION = "store-sales-time-series-forecasting"


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)

    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DATA_DIR)],
            check=True,
        )
        print(f"Dataset downloaded to {RAW_DATA_DIR}")
    except subprocess.CalledProcessError as e:
        print("Kaggle download failed. Ensure you have:")
        print("  1. Kaggle API credentials at ~/.kaggle/kaggle.json")
        print("  2. Accepted the competition rules at kaggle.com/competitions/store-sales-time-series-forecasting")
        sys.exit(1)
    except FileNotFoundError:
        print("kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)


if __name__ == "__main__":
    main()

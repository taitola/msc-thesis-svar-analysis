"""Download the project's data from FRED and store it as CSV files."""

import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from src import config


def get_fred_client():
    """Create a FRED client using FRED_API_KEY from the environment or .env file."""
    load_dotenv(config.ENV_PATH)
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        raise ValueError("FRED_API_KEY not found in environment variables.")
    return Fred(api_key=fred_key)


def fetch_fred_series(fred, ticker, start, end, freq=None):
    """Fetch one series, optionally resampled to `freq`. Returns None on failure."""
    try:
        series = fred.get_series(ticker, observation_start=start, observation_end=end)
        if freq:
            series = series.resample(freq).ffill()
            # Ensure full range coverage
            full_range = pd.date_range(start=start, end=end, freq=freq)
            series = series.reindex(full_range, method="ffill")
        return series
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def _download_national_series(fred, var_data, visuals_data):
    print("Processing National Data...")
    for name, ticker in config.NATIONAL_TICKERS.items():
        if name in config.NATIONAL_VISUALS_SERIES:
            start = (
                config.LONG_HISTORY_START_DATE
                if name in config.LONG_HISTORY_SERIES
                else config.VISUALS_START_DATE
            )
            series = fetch_fred_series(fred, ticker, start, config.VISUALS_END_DATE)
            if series is not None:
                visuals_data[f"National_{name}"] = series
                print(f"  Fetched {name} (Visuals)")
        else:
            series = fetch_fred_series(
                fred, ticker, config.VAR_START_DATE, config.VAR_END_DATE
            )
            if series is not None:
                var_data[name] = series
                print(f"  Fetched {name} (VAR)")


def _download_metro_series(fred, var_data, visuals_data):
    print("\nProcessing Metropolitan Data...")
    for city, tickers in config.METRO_TICKERS.items():
        for var_name, ticker in tickers.items():
            col_name = city.lower() if var_name == "hpi" else f"{city}_{var_name}"

            if var_name in ("permits", "pop"):
                # Population is annual and needs resampling to monthly
                freq = "MS" if var_name == "pop" else None
                series = fetch_fred_series(
                    fred,
                    ticker,
                    config.VISUALS_START_DATE,
                    config.VISUALS_END_DATE,
                    freq=freq,
                )
                if series is not None:
                    visuals_data[col_name] = series
                    print(f"  Fetched {col_name} (Visuals)")
            else:
                series = fetch_fred_series(
                    fred, ticker, config.VAR_START_DATE, config.VAR_END_DATE
                )
                if series is not None:
                    var_data[col_name] = series
                    print(f"  Fetched {col_name} (VAR)")


def download_data(fred):
    """Download all series and return them as (df_var, df_visuals)."""
    print("Starting Data Download...")
    var_data = {}
    visuals_data = {}

    _download_national_series(fred, var_data, visuals_data)
    _download_metro_series(fred, var_data, visuals_data)

    df_var = pd.DataFrame(var_data)
    df_visuals = pd.DataFrame(visuals_data)

    # FRED reports national permits in thousands
    if "National_permits" in df_visuals.columns:
        df_visuals["National_permits"] *= 1000

    return df_var, df_visuals


def save_data(df_var, df_visuals):
    """Write the VAR and exploratory datasets to the project's CSV files."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_var.to_csv(config.VAR_DATA_PATH)
    df_visuals.to_csv(config.VISUALS_DATA_PATH)
    print(f"\nSaved VAR data to {config.VAR_DATA_PATH}")
    print(f"Saved Visuals data to {config.VISUALS_DATA_PATH}")


def data_files_exist():
    """Return True if both CSV datasets have already been downloaded."""
    return config.VAR_DATA_PATH.exists() and config.VISUALS_DATA_PATH.exists()


def download_and_save():
    """Download all series from FRED and overwrite the local CSV files."""
    df_var, df_visuals = download_data(get_fred_client())
    save_data(df_var, df_visuals)

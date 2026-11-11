"""Load the CSV datasets and derive the variables used in plots and models."""

from collections.abc import Sequence
from os import PathLike

import numpy as np
import pandas as pd

from src import config


def load_data(path: str | PathLike) -> pd.DataFrame:
    """Load a CSV dataset with a parsed datetime index."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def prepare_var_data(df_var: pd.DataFrame, cs_cols: Sequence[str] = config.CS_COLS) -> pd.DataFrame:
    """Return the VAR dataset with model column names and log(hpi_t) for each index.

    Args:
        df_var: Raw VAR dataset as stored in `var_data.csv`.
        cs_cols: Case-Shiller columns, using model names, to transform.

    Returns:
        A copy of the data with a `log_<name>` column for each available index, so each
        model can form y_t = [unemp_t, pce_rate_t, ffr_t, log(hpi_t)]'.
    """
    df = df_var.rename(columns=config.VAR_COLUMN_RENAMES)
    df.index.name = "Date"

    for col in cs_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log(df[col])
        else:
            print(f"Warning: Column '{col}' not found in dataframe.")

    return df


def compute_permits_per_capita(
    df_visuals: pd.DataFrame,
    cities: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return monthly building permits per 1,000 residents, nationally and by metro.

    FRED reports population in thousands, so permits divided by population is a rate
    per 1,000 residents.

    Args:
        df_visuals: Exploratory dataset as stored in `visual_data.csv`.
        cities: Metro areas to include. Defaults to every metro in `config.METRO_TICKERS`.

    Returns:
        A DataFrame with a `National` column and one column per available metro.
    """
    if cities is None:
        cities = list(config.METRO_TICKERS)

    df_per_capita = pd.DataFrame(index=df_visuals.index)

    # National permits are a seasonally adjusted annual rate, so divide by 12 for a monthly rate
    if {"National_permits", "National_population"} <= set(df_visuals.columns):
        df_per_capita["National"] = (
            df_visuals["National_permits"] / df_visuals["National_population"]
        ) / 12

    for city in cities:
        permits_col = f"{city}_permits"
        pop_col = f"{city}_pop"
        if permits_col in df_visuals.columns and pop_col in df_visuals.columns:
            # Annual metro population is forward-filled across the 12 months of each year
            # during download. The fills below ensure complete monthly coverage, including
            # the months before the first observation in the combined monthly index.
            pop_series = df_visuals[pop_col].interpolate(method="linear").ffill().bfill()
            df_per_capita[city] = df_visuals[permits_col] / pop_series

    return df_per_capita

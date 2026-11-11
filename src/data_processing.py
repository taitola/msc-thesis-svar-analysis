"""Load the CSV datasets and derive the variables used in plots and models."""

import numpy as np
import pandas as pd

from src import config


def load_data(path):
    """Load a CSV dataset with a parsed datetime index."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def prepare_var_data(df_var, cs_cols=config.CS_COLS):
    """Return the VAR dataset with model column names and log house price indices."""
    df = df_var.rename(columns=config.VAR_COLUMN_RENAMES)
    df.index.name = "Date"

    for col in cs_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log(df[col])
        else:
            print(f"Warning: Column '{col}' not found in dataframe.")

    return df


def compute_permits_per_capita(df_visuals, cities=None):
    """Return building permits divided by population, nationally and for each metro."""
    if cities is None:
        cities = list(config.METRO_TICKERS)

    df_per_capita = pd.DataFrame(index=df_visuals.index)

    # National permits are an annual rate, so divide by 12 for a monthly rate
    if {"National_permits", "National_population"} <= set(df_visuals.columns):
        df_per_capita["National"] = (
            df_visuals["National_permits"] / df_visuals["National_population"]
        ) / 12

    for city in cities:
        permits_col = f"{city}_permits"
        pop_col = f"{city}_pop"
        if permits_col in df_visuals.columns and pop_col in df_visuals.columns:
            pop_series = df_visuals[pop_col].interpolate(method="linear").ffill().bfill()
            df_per_capita[city] = df_visuals[permits_col] / pop_series

    return df_per_capita

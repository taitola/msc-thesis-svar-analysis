"""Simulate empirical counterfactual trajectories by zeroing structural FFR shocks.

Historical decompositions for stationary models do not apply to I(1) variables in levels,
because the initial-condition transients do not die out. Following Kilian & Lütkepohl
(2017, p. 120), the paths are simulated in-sample conditional on

    y_t^{cf} = nu + A_1 y_{t-1}^{cf} + ... + A_12 y_{t-12}^{cf} + B_0^{-1} w_t^{cf}

where w_t^{cf} = w_t for t < T* and w_t^{cf} = (w_1t, w_2t, 0, w_4t)' for t >= T*.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults

from src import config


def _simulate_counterfactual_path(
    results: VARResults,
    ffr_idx: int,
    cf_start_date: str,
) -> pd.DataFrame:
    """
    Replicates Equation (5.18) from the thesis:
      1. Orthogonalizes reduced-form residuals into structural shocks: w_t = B_0 u_t.
      2. Sets structural monetary policy shocks w_{ffr, t} = 0 for t >= cf_start_date.
      3. Propagates counterfactual endogenous variables y_t^{cf} dynamically forward
         using the counterfactual lagged realizations y_{t-1}^{cf}, ..., y_{t-p}^{cf}.

    Returns:
        DataFrame of y_t^{cf} with one column per model variable, over the full sample.
    """
    p = results.k_ar
    T, K = results.resid.shape
    endog = results.model.endog              # Shape: (p + T, K)
    dates = results.model.data.dates         # DatetimeIndex of length p + T
    resid_dates = results.resid.index        # DatetimeIndex of length T

    # Cholesky factor B_0^{-1} = P with Stata 1/T ML scaling
    df_scale = np.sqrt(results.df_resid / results.nobs)
    P = np.linalg.cholesky(results.sigma_u) * df_scale
    P_inv = np.linalg.inv(P)

    # Recover historical structural shocks: W = U * (P^{-1})'
    resid_values = results.resid.values
    W = resid_values @ P_inv.T

    # Force structural FFR shocks (w_{3t}) to zero from T* (2001-01-01) onward
    W_cf = W.copy()
    cf_resid_mask = resid_dates >= pd.Timestamp(cf_start_date)
    W_cf[cf_resid_mask, ffr_idx] = 0.0

    # Map modified structural shocks back to reduced-form innovations: U^{cf} = W^{cf} * P'
    U_cf = W_cf @ P.T

    # Recursively simulate y_t^{cf} = v + A_1 y_{t-1}^{cf} + ... + A_p y_{t-p}^{cf} + u_t^{cf}
    intercept = results.intercept
    coefs = results.coefs
    y_cf = endog.copy()

    t_start = int(np.searchsorted(dates, pd.Timestamp(cf_start_date)))
    t_start = max(t_start, p)

    for t in range(t_start, p + T):
        lags_reversed = y_cf[t - p:t][::-1]
        y_cf[t] = intercept + np.einsum('pij,pj->i', coefs, lags_reversed) + U_cf[t - p]

    return pd.DataFrame(y_cf, index=dates, columns=results.names)


def run_3var_counterfactual(
    df: pd.DataFrame,
    start_date: str = config.VAR_START_DATE,
    end_date: str = config.VAR_END_DATE,
    cf_start_date: str = config.COUNTERFACTUAL_START_DATE,
    lags: int = config.VAR_LAGS,
) -> pd.DataFrame:
    """
    Simulates the baseline 3-variable VAR counterfactual (unemp, pce_rate, ffr)
    omitting house prices, used as the benchmark in Figure 21.

    Returns:
        DataFrame with columns ['actual_ffr', 'cf_ffr_3var'].
    """
    data = df.copy()
    if start_date is not None:
        data = data.loc[data.index >= start_date]
    if end_date is not None:
        data = data.loc[data.index <= end_date]

    vars_3 = ['unemp', 'pce_rate', 'ffr']
    data_subset = data[vars_3].dropna()
    data_subset = data_subset.set_axis(pd.DatetimeIndex(data_subset.index, freq='MS'))

    results = VAR(data_subset).fit(lags)
    df_cf = _simulate_counterfactual_path(results, ffr_idx=2, cf_start_date=cf_start_date)

    return pd.DataFrame({
        'actual_ffr': data_subset['ffr'],
        'cf_ffr_3var': df_cf['ffr'],
    })


def run_counterfactual_analysis(
    df: pd.DataFrame,
    cs_cols: list[str] = config.CS_COLS,
    start_date: str = config.VAR_START_DATE,
    end_date: str = config.VAR_END_DATE,
    cf_start_date: str = config.COUNTERFACTUAL_START_DATE,
    lags: int = config.VAR_LAGS,
) -> dict[str, pd.DataFrame]:
    """
    Runs the 4-variable SVAR empirical counterfactual simulation for each
    Case-Shiller index (setting w_{ffr, t} = 0 from 2001-01-01 onward).

    The printed Peak Diff is (actual peak - counterfactual peak) / counterfactual peak.

    Returns:
        dict: Mapping each cs_col to a DataFrame with columns
              ['actual_ffr', 'cf_ffr', 'actual_hpi', 'cf_hpi'].
    """
    cf_results = {}

    data = df.copy()
    if start_date is not None:
        data = data.loc[data.index >= start_date]
    if end_date is not None:
        data = data.loc[data.index <= end_date]

    print(f"Counterfactual Simulation: Structural FFR Shock = 0 from {cf_start_date}")
    print(f"{'Model':<12} | {'Actual Peak HPI':<16} | {'CF Peak HPI':<14} | {'Peak Diff (%)':<14}")
    print("-" * 64)

    for cs_col in cs_cols:
        current_vars = ['unemp', 'pce_rate', 'ffr', f'log_{cs_col}']
        data_subset = data[current_vars].dropna()

        if data_subset.empty:
            print(f"Warning: No data for {cs_col} in the specified range.")
            continue

        data_subset = data_subset.set_axis(pd.DatetimeIndex(data_subset.index, freq='MS'))
        results = VAR(data_subset).fit(lags)

        df_cf = _simulate_counterfactual_path(
            results, ffr_idx=2, cf_start_date=cf_start_date
        )

        # Convert log Case-Shiller series back to index levels (2000/01 = 100)
        actual_hpi = np.exp(data_subset[f'log_{cs_col}'])
        cf_hpi = np.exp(df_cf[f'log_{cs_col}'])

        peak_actual = actual_hpi.max()
        peak_cf = cf_hpi.max()
        pct_diff = ((peak_actual - peak_cf) / peak_cf) * 100

        print(f"{cs_col:<12} | {peak_actual:<16.2f} | {peak_cf:<14.2f} | {pct_diff:<+14.2f}%")

        cf_results[cs_col] = pd.DataFrame({
            'actual_ffr': data_subset['ffr'],
            'cf_ffr': df_cf['ffr'],
            'actual_hpi': actual_hpi,
            'cf_hpi': cf_hpi,
        })

    return cf_results
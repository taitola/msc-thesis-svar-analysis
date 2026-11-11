"""Estimate Cholesky-identified VAR(12) models and Stata-compatible impulse responses.

Each model is the reduced form (Kilian & Lütkepohl, 2017, Eq. 2.2.4)

    y_t = nu + A_1 y_{t-1} + ... + A_12 y_{t-12} + u_t,    u_t ~ (0, Sigma_u)

with y_t = [unemp_t, pce_rate_t, ffr_t, log(hpi_t)]'. Structural shocks satisfy
u_t = B_0^{-1} w_t, where B_0^{-1} = P = chol(Sigma_u) is lower triangular.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults

IrfBand = tuple[np.ndarray, np.ndarray, np.ndarray]


def _compute_stata_oirf(results: VARResults, steps: int) -> np.ndarray:
    """
    Computes orthogonalized IRFs Theta_i = J A^i J' B_0^{-1} for i = 0, ..., steps,
    matching Stata's default `var` ML covariance estimator.

    `statsmodels` estimates Sigma_u by dividing by T - Kp - 1, while Stata divides by T,
    so the Cholesky-based responses are scaled by sqrt((T - Kp - 1) / T).

    Returns:
        Array of shape (steps + 1, K, K), where element [i, 3, 2] is theta_{43,i}.
    """
    df_scale = np.sqrt(results.df_resid / results.nobs)
    return results.irf(steps).orth_irfs * df_scale


def _bootstrap_stata_oirf_se(
    results: VARResults,
    steps: int = 60,
    n_boot: int = 500,
    seed: int = 55555,
) -> np.ndarray:
    """
    Implements Stata's exact residual bootstrap algorithm (`irf create, bs reps(R)`)
    as documented in [TS] irf create (Methods and formulas, p. 26):
      1. Condition on the first p observations (Y_init).
      2. Resample T residual vectors (1 x K) with replacement.
      3. Recursively simulate synthetic sample Y* of length p + T.
      4. Re-estimate VAR(p), compute OIRFs (with 1/T ML scaling), and return
         the sample standard deviation (ddof=1) across R replications.

    Resampling whole residual vectors preserves contemporaneous cross-equation correlation.

    Returns:
        Array of shape (steps + 1, K, K) with the bootstrap standard error of each response.
    """
    rng = np.random.default_rng(seed)

    p = results.k_ar
    T, K = results.resid.shape
    endog = results.model.endog              # Shape: (p + T, K)
    y_init = endog[:p].copy()                # First p presample observations: (p, K)
    resid_values = results.resid.values      # Estimated residuals: (T, K)

    intercept = results.intercept            # Shape: (K,)
    coefs = results.coefs                    # Shape: (p, K, K) -> A_1, ..., A_p

    boot_oirfs = np.empty((n_boot, steps + 1, K, K))

    for r in range(n_boot):
        # Step 3a: Draw T row vectors with replacement from residuals
        sampled_idx = rng.integers(0, T, size=T)
        u_star = resid_values[sampled_idx]

        # Step 3b: Construct synthetic sample starting from the p initial observations
        y_star = np.empty((p + T, K))
        y_star[:p] = y_init

        for t in range(p, p + T):
            # y_t* = v + A_1 y_{t-1}* + ... + A_p y_{t-p}* + u_t*
            lags_reversed = y_star[t - p:t][::-1]  # [y_{t-1}*, ..., y_{t-p}*]
            y_star[t] = intercept + np.einsum('pij,pj->i', coefs, lags_reversed) + u_star[t - p]

        # Step 3c & 3d: Fit VAR(p) on synthetic sample and store OIRFs
        boot_res = VAR(y_star).fit(p)
        boot_oirfs[r] = _compute_stata_oirf(boot_res, steps)

    # Step 4: Standard error is the sample standard deviation across R replications
    return np.std(boot_oirfs, axis=0, ddof=1)


def run_var_analysis(
    df: pd.DataFrame,
    cs_cols: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    lags: int = 12,
    irf_steps: int = 60,
    n_boot: int = 500,
    seed: int = 55555,
) -> dict[str, IrfBand]:
    """
    Fits 4-variable SVAR models for each Case-Shiller index and computes
    orthogonalized IRFs with symmetric bootstrapped 95% confidence intervals
    matching Stata's `irf create, step(60) bs reps(500)`.

    Args:
        df: Output of `prepare_var_data`.
        cs_cols: Case-Shiller indices to model, one VAR each.
        start_date: First month of the sample, including the p presample observations.
        end_date: Last month of the sample.
        lags: Lag order p.
        irf_steps: Maximum horizon i.
        n_boot: Number of bootstrap replications R.
        seed: Seed for the NumPy random generator used in each model's bootstrap.

    Returns:
        Mapping from index name to (response, lower, upper) for theta_{43,i}, the response
        of log(hpi_{t+i}) to a 1-SD structural FFR shock w_3t, with bands
        theta_{43,i} +/- 1.96 * SE_bs.
    """
    irf_results = {}
    z_crit = norm.ppf(0.975)  # ~1.959964 (matches Stata's 95% normal CI)

    data = df.copy()
    if start_date is not None:
        data = data.loc[data.index >= start_date]
    if end_date is not None:
        data = data.loc[data.index <= end_date]

    print(f"Sample Window (including {lags} lags): "
          f"{data.index.min().date()} to {data.index.max().date()}")
    print(f"{'Model':<12} | {'Eff. Obs (T)':<12} | {'FFR Residual SD':<16} | {'FFR Cholesky Shock':<18}")
    print("-" * 66)

    for cs_col in cs_cols:
        # Cholesky ordering: unemp, pce_rate, ffr, log(cs_col)
        current_vars = ['unemp', 'pce_rate', 'ffr', f'log_{cs_col}']
        data_subset = data[current_vars].dropna()

        if data_subset.empty:
            print(f"Warning: No data for {cs_col} in the specified range.")
            continue

        # Declare the monthly frequency; this raises an error if any month is missing
        data_subset = data_subset.set_axis(pd.DatetimeIndex(data_subset.index, freq='MS'))

        results = VAR(data_subset).fit(lags)
        ffr_idx = 2  # 'ffr'
        cs_idx = 3   # f'log_{cs_col}'

        # Stata ML scaling factor: sqrt((T - Kp - 1) / T)
        df_scale = np.sqrt(results.df_resid / results.nobs)
        ffr_resid_std = results.resid['ffr'].std(ddof=1)
        P_stata = np.linalg.cholesky(results.sigma_u) * df_scale
        ffr_shock_size = P_stata[ffr_idx, ffr_idx]

        print(f"{cs_col:<12} | {results.nobs:<12} | {ffr_resid_std:<16.7f} | {ffr_shock_size:<18.8f}")

        # Point estimates (OIRF) and bootstrapped standard errors (stdoirf)
        oirf_matrix = _compute_stata_oirf(results, irf_steps)
        stdoirf_matrix = _bootstrap_stata_oirf_se(
            results, steps=irf_steps, n_boot=n_boot, seed=seed
        )

        response = oirf_matrix[:, cs_idx, ffr_idx]
        stderr = stdoirf_matrix[:, cs_idx, ffr_idx]

        # Symmetric 95% confidence intervals
        lower = response - z_crit * stderr
        upper = response + z_crit * stderr

        irf_results[cs_col] = (response, lower, upper)

    return irf_results

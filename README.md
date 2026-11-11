# Monetary Policy and the 2007 US Housing Bubble

**A Structural Vector Autoregressive Analysis in Python, cross-validated against Stata 19**

> Aitola, T. (2026). *Monetary Policy and the 2007 US Housing Bubble: A Structural Vector Autoregressive Analysis*. Master's Thesis in Economics, Tampere University.

Did the Federal Reserve's accommodative policy in 2001–2004 fuel the US housing bubble? This repository estimates separate 4-variable structural VARs on monthly data from 1987/01 to 2007/07 for two national house price aggregates and five metropolitan areas with very different housing supply elasticities. A one standard deviation monetary policy shock has large, persistent effects on house prices nationally and in supply-inelastic New York, and almost none in elastic Charlotte and Denver. A counterfactual that removes policy shocks from 2001 onward implies that the National, 10-City and New York peaks would have been 17–20% lower, while Las Vegas keeps its mid-2003 acceleration, which is consistent with a credit supply and speculation story.

**[Read the full walkthrough in `thesis_analysis.ipynb`](thesis_analysis.ipynb)**, with equations, figures and interpretation.

## Key Empirical Findings

| Series | Supply Elasticity / Market Regime | Construction Volume¹ | OIRF Significance & Trough² | Counterfactual Peak Impact³ |
|---|---|---|---|---|
| **National** | Aggregate | 0.51 | Significant, −1.95% at 57 months | +20.0% (184.6 vs 153.8) |
| **10-City** | Aggregate of large metros | Not computed | Significant, −2.26% at 60 months | +25.7% (226.9 vs 180.5) |
| **New York** | Inelastic, heavily regulated | 0.24, low | Significant, −1.84% at 58 months | +20.7% (216.6 vs 179.5) |
| **Las Vegas** | Elastic land, speculative boom | 1.69, very high | Significant, −2.49% at 48 months | +23.3% (235.8 vs 191.3), mid-2003 kink persists |
| **Denver** | Elastic, lightly regulated | 0.73, high | Not significant, −0.26% at 40 months | +3.5% (139.3 vs 134.7) |
| **Charlotte** | Elastic, lightly regulated | 1.23, high | Not significant, never below zero | −0.2% (133.7 vs 133.9) |
| **Cleveland** | Stagnating Rust Belt market | 0.26, low | Significant but small, −0.44% at 33 months | +3.0% (123.4 vs 119.8) |

¹ Building permits per 1,000 residents per month, 2000–2007 mean.
² Trough response of log house prices ×100 to a 1-SD FFR shock of about 12 basis points. Significant means the 95% bootstrap band lies below zero at the trough.
³ Actual peak relative to the counterfactual peak with policy shocks set to zero from 2001/01. Index values are actual vs counterfactual, with 2000/01 = 100.

## Methodology & Stata 19 Cross-Validation

**Model.** Each series has its own reduced-form VAR(12) in levels with $y_t = [\text{unemp}_t, \text{pce}_t, \text{ffr}_t, \log(\text{hpi}_t)]'$, following Kilian & Lütkepohl (2017). With 247 monthly observations and 12 lags, the effective sample is $T = 235$.

**Identification.** A recursive Cholesky ordering embeds a backward-looking Taylor Rule. The Fed responds within the month to unemployment and inflation, those variables respond to policy with a lag, and house prices respond immediately to all three.

**Counterfactual.** The simulation sets the structural policy shock to zero from 2001/01 and recursively rebuilds each series in-sample, following Kilian & Lütkepohl (2017, p. 120).

**Replicating Stata in Python.** [`src/var_irf.py`](src/var_irf.py) reproduces two Stata defaults that differ from `statsmodels`:

- **ML covariance scaling.** `statsmodels` divides the residual covariance by $T - Kp - 1$, and Stata's `var` divides by $T$. The code multiplies the Cholesky factor by $\sqrt{(T - Kp - 1)/T} = \sqrt{186/235} \approx 0.88966$.
- **Residual bootstrap.** The code implements the algorithm in Stata's *[TS] irf create* manual: 500 replications that condition on the presample, resample residual vectors, simulate recursively, re-estimate, and take the sample standard deviation across replications.

NumPy and Stata use different random number generators, so bootstrap standard errors match Stata up to Monte Carlo error rather than digit for digit.

## Repository Structure

```
.
├── thesis_analysis.ipynb    # Narrative walkthrough with equations, figures and results
├── data_csv/
│   ├── var_data.csv         # Model data, 1987/01–2007/07
│   └── visual_data.csv      # Exploratory data, 1959/01–2025/08
├── src/
│   ├── config.py            # Paths, sample periods, FRED tickers and model settings
│   ├── fred_download.py     # Optional data refresh from FRED
│   ├── data_processing.py   # Loading, log transforms and permits per capita
│   ├── var_irf.py           # VAR estimation and Stata-compatible bootstrapped IRFs
│   ├── counterfactual.py    # Counterfactual simulation with policy shocks set to zero
│   └── plots.py             # All figures, with optional saving at 300 dpi
├── requirements.txt
└── LICENSE
```

## Quickstart

`data_csv/var_data.csv` and `data_csv/visual_data.csv` are tracked in this repository, so you can clone, install and run the notebook immediately **without an API key**.

```bash
git clone <repository-url>
cd msc-thesis-data-platform
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open `thesis_analysis.ipynb` in VS Code with the Jupyter extension, select the `.venv` kernel and run all cells. The full notebook, including 3,500 bootstrap VAR estimations, takes about a minute on a typical laptop. The bootstrap uses a fixed seed, so repeated runs give the same results.

**Optional: fetch live data from FRED.** Create a `.env` file in the project root with your [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html), then set `REFRESH_DATA = True` in the notebook. This overwrites the CSV files, and later data revisions on FRED can change the results.

```
FRED_API_KEY=your_api_key_here
```

## License

Released under the [MIT License](LICENSE).

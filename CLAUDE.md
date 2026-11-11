You are refactoring and documenting a quantitative macroeconomics and econometrics portfolio repository accompanying my Master's Thesis in Economics: "Monetary Policy and the 2007 US Housing Bubble: A Structural Vector Autoregressive Analysis" (Tampere University, 2026).

Please read all files in `src/`, `thesis_analysis.ipynb`, `README.md`, and `requirements.txt`. Do NOT alter the core numerical logic that replicates Stata 19's exact point estimates and bootstrap standard errors, but refactor the codebase for maximum readability, polish the plots, and transform `thesis_analysis.ipynb` and `README.md` into a publication-grade portfolio showcase.

---

### I. THEORETICAL & ECONOMETRIC CONTEXT (USE THIS IN MARKDOWN & DOCSTRINGS)

#### 1. Research Question & Economic Framework
- **Core Question:** Did the Federal Reserve's accommodative monetary policy (deviations of the Federal Funds Rate from the Taylor Rule during 2001–2004) fuel the 2000s U.S. housing bubble, and how did transmission vary across metropolitan areas (MSAs) with different housing supply elasticities?
- **Sample Window:** Monthly data from `1987-01-01` to `2007-07-01` (N = 247 observations; with p = 12 lags, T = 235 effective observations from `1988-01` to `2007-07`). The sample begins with the Great Moderation / Greenspan era and Case-Shiller inception, and ends in July 2007 just prior to the August 2007 GFC outbreak.
- **Monetary Transmission Channels:**
  1. *Interest Rate Channel (Neoclassical):* Operates through the user cost of housing capital (Himmelberg et al., 2005), acting as the risk-free discount factor (r) for durable housing assets and altering fundamental house values.
  2. *Bank Lending / Credit Supply Channel (Non-Neoclassical):* Operates through financial frictions, intermediary balance sheets, and external finance premiums (Disyatat, 2011; Brunnermeier, 2009). After the FOMC cut the FFR to 1% in June 2003, the GSE refinancing boom collapsed and mortgage originators pivoted to non-conforming/subprime loans funded via Private-Label Mortgage-Backed Securities (PLS) and shadow banking wholesale funding (Justiniano et al., 2022; Drechsler et al., 2022), fueling speculative buying by a homogeneous marginal buyer pool (Mian & Sufi, 2022).
- **Regional Supply Elasticity & Bubble Dynamics (Glaeser, Gyourko & Saiz, 2008):**
  We estimate separate 4-variable SVARs for 2 national aggregates (`National` CSUSHPISA, `10-City` SPCS10RSA) and 5 categorical MSAs:
  - **New York (`NYXRSA`):** Heavily regulated, inelastic housing supply, low building permits per capita, low speculative share. Demand shocks cannot be absorbed by construction, leading to strong, persistent fundamental price responses to FFR shocks.
  - **Charlotte (`CRXRSA`) & Denver (`DNXRSA`):** Lightly regulated, elastic housing supply, high building permits per capita. Increased housing demand is met by new construction rather than price appreciation, resulting in statistically insignificant IRFs.
  - **Cleveland (`CEXRSA`):** Stagnating Rust Belt market with excess housing stock and house prices below minimum profitable construction replacement costs. Minimal new construction; small but statistically significant IRF tied to national macro fundamentals.
  - **Las Vegas (`LVXRSA`):** Elastic land supply but experienced simultaneous explosive construction and price booms starting in mid-2003, driven by the PLS credit supply surge and extreme speculative activity (non-owner-occupied purchases). Exhibits a large IRF, but high residual volatility (ARCH effects) and lack of cointegration indicate decoupling from macroeconomic fundamentals.

#### 2. Econometric Specification (Kilian & Lütkepohl, 2017)
- **Reduced-Form VAR(p) in Levels (Kilian & Lütkepohl, 2017, Ch. 2, Eq. 2.2.4):**
  $$y_t = \nu + A_1 y_{t-1} + \dots + A_{12} y_{t-12} + u_t, \quad u_t \sim (0, \Sigma_u)$$
  where $K = 4$, $p = 12$, and $y_t = [\text{unemp}_t, \text{pce\_rate}_t, \text{ffr}_t, \log(\text{hpi}_t)]'$.
  - Estimated in levels via OLS: Even with I(1) variables and cointegration, OLS/ML slope estimators remain consistent and asymptotically normal for $p > 1$ (Sims, Stock & Watson, 1990; K&L Section 2.3.5, pp. 41–42), avoiding inconsistency from over-differencing or misspecified VECM cointegration rank.
  - Fixed lag order $p = 12$: Captures a full 1-year monetary transmission window and avoids post-model-selection inference distortions (Leeb & Pötscher, 2005; K&L Section 2.6.5, pp. 61–64).
- **ML vs. OLS Innovation Covariance Scaling (K&L Section 2.3.1 Eq. 2.3.4 vs. Section 2.3.4 p. 40):**
  - Python's `statsmodels` computes the OLS covariance estimator: $\hat{\Sigma}_u = \frac{1}{T - Kp - 1} \hat{U}\hat{U}'$ (dividing by `df_resid` = 235 - 49 = 186).
  - Stata's `var` command defaults to the Gaussian Maximum Likelihood (ML) estimator: $\tilde{\Sigma}_u = \frac{1}{T} \hat{U}\hat{U}' = \frac{T - Kp - 1}{T} \hat{\Sigma}_u$ (dividing by $T = 235$).
  - Multiplying the `statsmodels` Cholesky factor by $\sqrt{(T - Kp - 1)/T} = \sqrt{186/235} \approx 0.88966$ achieves 1:1 numerical parity with Stata's 1-SD structural FFR shock ($\approx 12.34$ basis points).
- **Recursive Cholesky Identification (Christiano, Eichenbaum & Evans, 1999, 2005; K&L Ch. 4, pp. 109–112):**
  - Structural shocks $w_t$ with $\mathbb{E}(w_t w_t') = I_K$ are related to reduced-form innovations via $u_t = B_0^{-1} w_t$, where $B_0^{-1} = P = \text{chol}(\tilde{\Sigma}_u)$ is lower-triangular:
    $$\begin{bmatrix} u_t^{\text{unemp}} \\ u_t^{\text{pce}} \\ u_t^{\text{ffr}} \\ u_t^{\text{hpi}} \end{bmatrix} = \begin{bmatrix} b_{11} & 0 & 0 & 0 \\ b_{21} & b_{22} & 0 & 0 \\ b_{31} & b_{32} & b_{33} & 0 \\ b_{41} & b_{42} & b_{43} & b_{44} \end{bmatrix} \begin{bmatrix} w_{1t} \\ w_{2t} \\ w_{3t} \\ w_{4t} \end{bmatrix}$$
  - Economic interpretation: Ordering `unemp` and `pce_rate` before `ffr` embeds a backward-looking Taylor Rule reaction function. The Fed responds contemporaneously (within the month) to unemployment and core inflation, while macroeconomic aggregates respond to monetary policy with at least a one-month lag. Ordering `log(hpi)` last assumes house prices respond contemporaneously to all three macro variables, while the Fed does not target house prices contemporaneously ($b_{34} = 0$). By Christiano et al. (2005), the impulse responses to $w_{3t}$ (the structural monetary policy shock) are invariant to the ordering of variables preceding `ffr`.
- **Companion Matrix & Structural Impulse Responses (K&L Section 4.1, pp. 110–112):**
  - Using the $Kp \times Kp$ ($48 \times 48$) companion matrix $A$ and selection matrix $J = [I_K, 0_{K \times K(p-1)}]$, reduced-form VMA weights are $\Phi_i = J A^i J'$ and orthogonalized structural impulse responses are $\Theta_i = \Phi_i B_0^{-1}$ for horizon $i = 0, 1, \dots, 60$. The response of log house prices to a 1-SD structural FFR shock is $\theta_{43, i} = \frac{\partial \log(\text{hpi}_{t+i})}{\partial w_{3t}}$.
- **Residual Bootstrap Confidence Intervals (Stata `[TS] irf create` Manual, p. 26):**
  - Condition on the $p = 12$ initial presample observations (`1987-01` to `1987-12`), resample $T = 235$ residual vectors ($1 \times K$) with replacement $R = 500$ times (`seed = 55555`) to preserve contemporaneous cross-equation correlation, recursively simulate synthetic samples $y_t^*$, re-estimate the VAR(12) and ML-scaled OIRFs, and compute the sample standard deviation (`ddof = 1`) across the 500 replications to form symmetric 95% confidence bands: $\hat{\theta}_{43, i} \pm 1.96 \times \widehat{\text{SE}}_{\text{bs}}$.
- **Empirical Counterfactual Simulation (K&L Section 4.3, p. 120):**
  - Standard stationary historical decompositions cannot be applied to I(1) levels variables because initial condition transients do not die out. Following Kilian & Lütkepohl (2017, p. 120), we simulate conditional in-sample counterfactual trajectories by setting structural monetary policy shocks $w_{3t}^{cf} = 0$ from $T^* = \text{2001-01-01}$ through `2007-07-01`, while preserving historical shocks prior to 2001 and all non-policy structural shocks ($w_{1t}, w_{2t}, w_{4t}$):
    $$y_t^{cf} = \hat{\nu} + \sum_{i=1}^{12} \hat{A}_i y_{t-i}^{cf} + \hat{B}_0^{-1} w_t^{cf}, \quad w_t^{cf} = \begin{cases} w_t & \text{if } t < \text{2001-01-01} \\ (w_{1t}, w_{2t}, 0, w_{4t})' & \text{if } t \ge \text{2001-01-01} \end{cases}$$

---

### II. TASKS TO EXECUTE

#### Task 1: Code Readability & Polish in `src/`
1. **`src/plots.py`:**
   - In `plot_irf_grid`, because the 4x2 grid has 7 panels and deletes the 8th subplot (`axes_flat[7]`), `sharex=True` hides the x-axis tick labels on `axes_flat[5]` (the `Charlotte` panel at the bottom of the right column). Fix this so BOTH bottom panels (`Cleveland` at `axes_flat[6]` and `Charlotte` at `axes_flat[5]`) display x-axis tick numbers (`0, 20, 40, 60`) and the `'Months'` label.
   - Add an optional `save_path=None` argument to all plotting functions in `src/plots.py` so figures can be saved cleanly (e.g., `if save_path: fig.savefig(save_path, dpi=300, bbox_inches='tight')`) before `plt.show()`.
2. **`src/data_processing.py`:**
   - In `compute_permits_per_capita`, clarify the comment/logic for population: annual metro population from FRED is forward-filled across the 12 months of each year during download, so `.ffill().bfill()` ensures complete monthly coverage.
3. **Docstrings & Type Hints:**
   - Add clean, concise docstrings and type hints across `src/config.py`, `src/fred_download.py`, `src/data_processing.py`, `src/var_irf.py`, `src/counterfactual.py`, and `src/plots.py` referencing the exact mathematical notation above.

#### Task 2: Upgrade `thesis_analysis.ipynb` into a Self-Contained Narrative Walkthrough
Update the Markdown and Code cells in `thesis_analysis.ipynb` so a technical recruiter, economist, or quant can read it top-to-bottom in GitHub's browser viewer and understand both the code and the economics:
1. **Introduction & Setup:** Briefly state the thesis objective, sample period (`1987/01–2007/07`), and note that pre-fetched datasets are included in `data_csv/` for immediate execution (`REFRESH_DATA = False` by default).
2. **Section 1: Data Ingestion & Preparation:** Load `visual_data.csv` and `var_data.csv`, and run `df_var_ready = prepare_var_data(df_var)`.
3. **Section 2: Exploratory Data Analysis & Regional Supply Elasticity:**
   - Provide concise economic context before each plot:
     - Housing Starts & Recessions (`1959–2025`): Housing construction as a leading business cycle indicator (Leamer, 2015).
     - National & Metro Building Permits per Capita: Contrast high-elasticity construction booms (`Las Vegas`, `Charlotte`, `Denver`) against supply-inelastic `New York` and stagnating `Cleveland` (Glaeser & Gyourko, 2018).
     - Case-Shiller House Price Indices (`1987/01–2007/07`, `2000/01 = 100`) & Macroeconomic Indicators (`FFR`, `Trimmed Mean PCE`, `Unemployment Rate`).
4. **Section 3: Structural VAR Methodology & Orthogonalized Impulse Responses (Figure 22):**
   - Include a clear Markdown section with the LaTeX equations for the reduced-form VAR(12), the Cholesky matrix $B_0^{-1}$, the companion VMA representation $\Theta_i = J A^i J' B_0^{-1}$, the $1/T$ ML covariance scaling factor $\sqrt{(T - Kp - 1)/T}$, and the 500-draw residual bootstrap algorithm.
   - Run `run_var_analysis` and `plot_irf_grid`, and include a bulleted interpretation of the empirical results across the 7 indices.
5. **Section 4: Empirical Counterfactual Simulation — Taylor Rule Baseline (Figures 21 & 23):**
   - Include a Markdown explanation of the counterfactual recursion (Kilian & Lütkepohl, 2017, p. 120) setting $w_{3t}^{cf} = 0$ from `2001/01` onward.
   - Run `run_3var_counterfactual` and `run_counterfactual_analysis`, plot both the 3-VAR vs. 4-VAR FFR comparison (Figure 21) and the 7x2 counterfactual grid (Figure 23), and summarize the key takeaway (20–30% lower peak prices in `National`, `10-City`, and `New York`, no change in elastic `Charlotte`/`Denver`, and the persistent mid-2003 PLS/speculative kink in `Las Vegas`).
6. **Execute the notebook** (or provide a one-liner command) so all cell outputs and plots are rendered and saved cleanly.

#### Task 3: Rewrite `README.md` as an Executive Portfolio Landing Page
Structure `README.md` with:
1. **Title & Overview:** Clear title, link/citation to the MSc Thesis, and a 1-paragraph executive summary.
2. **Key Empirical Findings:** A clean summary table of the 7 Case-Shiller series (Supply Elasticity / Market Regime, Construction Volume, OIRF Significance & Trough, Counterfactual Peak Impact).
3. **Methodology & Stata 19 Cross-Validation:** Brief explanation of the 4-variable SVAR(12), Cholesky Taylor-rule identification, and how `src/var_irf.py` replicates Stata 19's exact ML covariance scaling ($1/T$) and 500-replication residual bootstrap.
4. **Repository Structure:** Clean file tree showing `thesis_analysis.ipynb`, `data_csv/`, and `src/`.
5. **Quickstart (Zero-Friction Execution):** Explicitly state that `data_csv/var_data.csv` and `data_csv/visual_data.csv` are tracked in the repository so users can clone, install `requirements.txt`, and run `thesis_analysis.ipynb` immediately without an API key, plus optional instructions for `.env` (`FRED_API_KEY`) if they wish to re-fetch live data from FRED.
# Master's Thesis Data Platform

This repository contains the data analysis and visualization code for a Master's Thesis ("gradu"). It studies how house prices in the United States respond to federal funds rate shocks, nationally and in selected metropolitan areas.

The analysis downloads economic data from FRED (Federal Reserve Economic Data), draws exploratory figures, and estimates vector autoregressions (VARs). Impulse responses and their bootstrapped confidence intervals replicate Stata's `irf create, step(60) bs reps(500)`.

## Project Structure

```
.
├── thesis_analysis.ipynb   # Runs the full analysis and shows the results
├── src/
│   ├── config.py           # Paths, sample periods, FRED tickers and model settings
│   ├── fred_download.py    # Downloads data from FRED and saves it as CSV
│   ├── data_processing.py  # Loads the CSVs and derives model and plot variables
│   ├── var_irf.py          # VAR estimation and Stata-style bootstrapped IRFs
│   └── plots.py            # Exploratory figures and the IRF grid
├── data_csv/               # Downloaded data (ignored by git)
├── requirements.txt        # Python dependencies
└── .env                    # FRED API key (ignored by git, you create this)
```

`thesis_analysis.ipynb` contains only function calls, explanatory text and the resulting figures. To change a ticker, date range or model setting, edit `src/config.py`.

## Setup

1.  **Clone the repository**

2.  **Set up the environment**

    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Add your FRED API key**

    Downloading the data requires a [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html). Create a `.env` file in the project root:
    ```
    FRED_API_KEY=your_api_key_here
    ```

## Usage

Open `thesis_analysis.ipynb`, select the environment as the kernel, and run all cells. Run it from the project root so that `from src import ...` resolves, which is the default when the notebook is opened from this folder.

The notebook has three parts:

1.  **Data:** downloads the data if `data_csv/` has no CSV files, then loads them. Set `REFRESH_DATA = True` to download everything again and overwrite the files.
2.  **Exploratory analysis:** permits per capita, Case-Shiller indices, economic indicators, and housing starts with recessions.
3.  **VAR analysis:** one 12-lag VAR per house price index, identified with the Cholesky ordering unemployment, PCE inflation, federal funds rate, log house prices. The results are plotted as a grid of impulse responses.

The bootstrap uses a fixed seed, so repeated runs on the same data give the same results.

## Dependencies

- pandas
- fredapi
- numpy
- scipy
- matplotlib
- seaborn
- statsmodels
- python-dotenv
- ipykernel

"""Paths, sample periods, FRED tickers and model settings used across the project.

Each structural VAR has K = 4 variables and p = 12 lags, with the Cholesky ordering
y_t = [unemp_t, pce_rate_t, ffr_t, log(hpi_t)]'. With N = 247 monthly observations from
1987-01 to 2007-07, the effective sample is T = N - p = 235 (1988-01 to 2007-07).
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_PATH: Path = PROJECT_ROOT / ".env"
DATA_DIR: Path = PROJECT_ROOT / "data_csv"
VAR_DATA_PATH: Path = DATA_DIR / "var_data.csv"
VISUALS_DATA_PATH: Path = DATA_DIR / "visual_data.csv"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"  # Saved key figures, used in README.md

# --- Sample periods ---
# VAR estimation sample: Great Moderation start to the month before the GFC outbreak
VAR_START_DATE: str = "1987-01-01"
VAR_END_DATE: str = "2007-07-01"

# Exploratory plots: 2000 to present (for broader context)
VISUALS_START_DATE: str = "2000-01-01"
VISUALS_END_DATE: str = "2025-08-01"

# Long-run history for housing starts and recessions
LONG_HISTORY_START_DATE: str = "1959-01-01"

# --- FRED tickers ---
NATIONAL_TICKERS: dict[str, str] = {
    # VAR series
    "cs": "CSUSHPISA",  # Case-Shiller National Home Price Index
    "cs10": "SPCS10RSA",  # Case-Shiller 10-City Home Price Index
    "ffr": "FEDFUNDS",  # Federal Funds Rate
    "unemp": "UNRATE",  # Unemployment Rate
    "pce_rate": "PCETRIM12M159SFRBDAL",  # Dallas Fed Trimmed Mean PCE, 12-month rate
    # Exploratory plot series
    "permits": "PERMIT",  # Building Permits
    "population": "POPTHM",  # Population (thousands)
    "housing starts": "HOUST",  # Housing Starts
    "recessions": "USREC",  # Recession Indicator
}

# National series that go to the exploratory dataset rather than the VAR dataset
NATIONAL_VISUALS_SERIES: tuple[str, ...] = ("permits", "population", "housing starts", "recessions")

# Exploratory series downloaded from LONG_HISTORY_START_DATE
LONG_HISTORY_SERIES: tuple[str, ...] = ("housing starts", "recessions")

# hpi: S&P/Case-Shiller Home Price Index (seasonally adjusted)
# permits: New Private Housing Units Authorized by Building Permits (seasonally adjusted)
# pop: Resident Population in thousands (annual, carried forward to monthly on download)
METRO_TICKERS: dict[str, dict[str, str]] = {
    "Charlotte": {"hpi": "CRXRSA", "permits": "CHAR737BPPRIVSA", "pop": "CGRPOP"},
    "Cleveland": {"hpi": "CEXRSA", "permits": "CLEV439BPPRIVSA", "pop": "CVLPOP"},
    "Denver": {"hpi": "DNXRSA", "permits": "DENV708BPPRIVSA", "pop": "DNVPOP"},
    "Las Vegas": {"hpi": "LVXRSA", "permits": "LASV832BPPRIVSA", "pop": "LSVPOP"},
    "New York": {"hpi": "NYXRSA", "permits": "NEWY636BPPRIVSA", "pop": "NYTPOP"},
}

# --- Exploratory plots ---
HPI_COLS: list[str] = ["cs", "cs10", "cleveland", "denver", "las vegas", "new york", "charlotte"]

ECONOMIC_INDICATORS: dict[str, str] = {
    "ffr": "Federal Funds Rate",
    "unemp": "Unemployment Rate",
    "pce_rate": "Trimmed Mean PCE Inflation (12-Month)",
}

# --- VAR model ---
VAR_COLUMN_RENAMES: dict[str, str] = {"new york": "newyork", "las vegas": "lasvegas"}

# Case-Shiller indices in the order they are estimated and plotted
CS_NAMES: dict[str, str] = {
    "cs": "National",
    "cs10": "10-City",
    "newyork": "New York",
    "lasvegas": "Las Vegas",
    "denver": "Denver",
    "charlotte": "Charlotte",
    "cleveland": "Cleveland",
}
CS_COLS: list[str] = list(CS_NAMES)

VAR_LAGS: int = 12  # p
IRF_STEPS: int = 60  # Maximum horizon i of Theta_i, in months
N_BOOTSTRAP: int = 500  # R, residual bootstrap replications
BOOTSTRAP_SEED: int = 55555

# --- Counterfactual simulation ---
# T*: first month in which the structural policy shock w_3t is set to zero
COUNTERFACTUAL_START_DATE: str = "2001-01-01"

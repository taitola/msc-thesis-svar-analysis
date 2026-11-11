"""Paths, sample periods, FRED tickers and model settings used across the project."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data_csv"
VAR_DATA_PATH = DATA_DIR / "var_data.csv"
VISUALS_DATA_PATH = DATA_DIR / "visual_data.csv"

# --- Sample periods ---
# VAR estimation sample: 1987 to mid-2007 (pre-crisis)
VAR_START_DATE = "1987-01-01"
VAR_END_DATE = "2007-07-01"

# Exploratory plots: 2000 to present (for broader context)
VISUALS_START_DATE = "2000-01-01"
VISUALS_END_DATE = "2025-08-01"

# Long-run history for housing starts and recessions
LONG_HISTORY_START_DATE = "1959-01-01"

# --- FRED tickers ---
NATIONAL_TICKERS = {
    # VAR series
    "cs": "CSUSHPINSA",  # Case-Shiller National Home Price Index
    "cs10": "SPCS10RSA",  # Case-Shiller 10-City Home Price Index
    "ffr": "FEDFUNDS",  # Federal Funds Rate
    "unemp": "UNRATE",  # Unemployment Rate
    "pce_rate": "PCETRIM12M159SFRBDAL",  # PCE 12-Month Rate of Change
    # Exploratory plot series
    "permits": "PERMIT",  # Building Permits
    "population": "POPTHM",  # Population (thousands)
    "housing starts": "HOUST",  # Housing Starts
    "recessions": "USREC",  # Recession Indicator
}

# National series that go to the exploratory dataset rather than the VAR dataset
NATIONAL_VISUALS_SERIES = ("permits", "population", "housing starts", "recessions")

# Exploratory series downloaded from LONG_HISTORY_START_DATE
LONG_HISTORY_SERIES = ("housing starts", "recessions")

# hpi: S&P/Case-Shiller Home Price Index (seasonally adjusted)
# permits: New Private Housing Units Authorized by Building Permits (seasonally adjusted)
# pop: Resident Population (annual, carried forward to monthly on download)
METRO_TICKERS = {
    "Charlotte": {"hpi": "CRXRSA", "permits": "CHAR737BPPRIVSA", "pop": "CGRPOP"},
    "Cleveland": {"hpi": "CEXRSA", "permits": "CLEV439BPPRIVSA", "pop": "CVLPOP"},
    "Denver": {"hpi": "DNXRSA", "permits": "DENV708BPPRIVSA", "pop": "DNVPOP"},
    "Las Vegas": {"hpi": "LVXRSA", "permits": "LASV832BPPRIVSA", "pop": "LSVPOP"},
    "New York": {"hpi": "NYXRSA", "permits": "NEWY636BPPRIVSA", "pop": "NYTPOP"},
}

# --- Exploratory plots ---
HPI_COLS = ["cs", "cs10", "cleveland", "denver", "las vegas", "new york", "charlotte"]

ECONOMIC_INDICATORS = {
    "ffr": "Federal Funds Rate",
    "unemp": "Unemployment Rate",
    "pce_rate": "PCE 12-Month Rate of Change",
}

# --- VAR model ---
VAR_COLUMN_RENAMES = {"new york": "newyork", "las vegas": "lasvegas"}

# Case-Shiller indices in the order they are estimated and plotted
CS_NAMES = {
    "cs": "National",
    "cs10": "10-City",
    "newyork": "New York",
    "lasvegas": "Las Vegas",
    "denver": "Denver",
    "charlotte": "Charlotte",
    "cleveland": "Cleveland",
}
CS_COLS = list(CS_NAMES)

VAR_LAGS = 12
IRF_STEPS = 60
N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 55555

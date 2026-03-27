"""Project configuration values."""

SECTOR_BENCHMARK_WEIGHTS = {
    "Information Technology": 0.307,
    "Financials": 0.145,
    "Health Care": 0.108,
    "Consumer Discretionary": 0.105,
    "Communication Services": 0.095,
    "Industrials": 0.083,
    "Consumer Staples": 0.059,
    "Energy": 0.033,
    "Utilities": 0.024,
    "Real Estate": 0.022,
    "Materials": 0.020,
}

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

TRAIN_START_DATE = "2019-01-01"
TRAIN_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2024-12-31"

SECTOR_BAND = 0.05
MIN_STOCKS_PER_SECTOR = 10
STOCKS_PER_SECTOR = 10

DOWNLOAD_BATCH_SIZE = 50
DOWNLOAD_RETRY_DELAY_SECONDS = 8
DOWNLOAD_BATCH_DELAY_SECONDS = 8
MAX_DOWNLOAD_RETRIES = 3

TRADING_DAYS_PER_YEAR = 252
SP500_BENCHMARK_2024 = 0.2502

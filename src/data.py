"""Data acquisition and preprocessing helpers."""

import time
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

from .config import (
    DOWNLOAD_BATCH_DELAY_SECONDS,
    DOWNLOAD_BATCH_SIZE,
    DOWNLOAD_RETRY_DELAY_SECONDS,
    MAX_DOWNLOAD_RETRIES,
    WIKI_SP500_URL,
)


def get_sp500_tickers_and_sectors() -> Tuple[List[str], Dict[str, str]]:
    """Scrape S&P 500 symbols and sector mapping from Wikipedia."""
    sp500_table = pd.read_html(WIKI_SP500_URL)[0]
    tickers = sp500_table["Symbol"].tolist()
    sectors = sp500_table.set_index("Symbol")["GICS Sector"].to_dict()
    return tickers, sectors


def download_data(
    tickers: List[str],
    start: str,
    end: str,
    max_retries: int = MAX_DOWNLOAD_RETRIES,
) -> pd.DataFrame:
    """Download close prices from Yahoo Finance in batches with retries."""
    all_data = {}
    for i in range(0, len(tickers), DOWNLOAD_BATCH_SIZE):
        batch = tickers[i : i + DOWNLOAD_BATCH_SIZE]
        retries = 0
        while retries < max_retries:
            try:
                batch_data = yf.download(batch, start=start, end=end)["Close"]
                all_data.update(batch_data.to_dict())
                break
            except Exception as error:  # noqa: BLE001
                print(f"Error downloading batch {batch}: {error}")
                retries += 1
                time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)
        time.sleep(DOWNLOAD_BATCH_DELAY_SECONDS)
    return pd.DataFrame(all_data)


def prepare_returns(price_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Drop tickers with missing data and compute daily returns."""
    cleaned_data = price_data.dropna(axis=1, how="any")
    filtered_tickers = cleaned_data.columns.tolist()
    returns = cleaned_data.pct_change().dropna()
    return cleaned_data, filtered_tickers, returns

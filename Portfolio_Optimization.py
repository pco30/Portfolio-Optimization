# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 04:14:15 2025

@author: PRINCELY OSEJI
"""

import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import time

# ---------------------------- STEP 1: SCRAPE S&P 500 & SECTOR DATA ---------------------------- #
# Scrape S&P 500 constituents
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_table = pd.read_html(url)[0]
tickers = sp500_table['Symbol'].tolist()
sectors = sp500_table.set_index('Symbol')['GICS Sector'].to_dict()

# Sector ETFs (for backtesting, if needed)
sector_etfs = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

# Sector benchmark weights (given values)
sector_benchmark_weights = {
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

# ---------------------------- STEP 2: DOWNLOAD STOCK DATA ---------------------------- #
def download_data(tickers, start, end, max_retries=3):
    all_data = {}
    batch_size = 50  # Reduce request load
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        retries = 0
        while retries < max_retries:
            try:
                batch_data = yf.download(batch, start=start, end=end)['Close']
                all_data.update(batch_data.to_dict())
                break  # Exit retry loop if successful
            except Exception as e:
                print(f"Error downloading batch {batch}: {e}")
                retries += 1
                time.sleep(8)  # Increase delay to prevent rate limit
        time.sleep(8)  # Add delay after each batch
    return pd.DataFrame(all_data)

# Download historical stock data (2019-2023 for training)
start_date = "2019-01-01"
end_date = "2023-12-31"
data = download_data(tickers, start_date, end_date)

# Drop tickers with missing data
data.dropna(axis=1, how='any', inplace=True)
filtered_tickers = data.columns.tolist()

# ---------------------------- STEP 3: SECTOR-LEVEL OPTIMIZATION ---------------------------- #
returns = data.pct_change().dropna()

# Optimization Variables
n_sectors = len(sector_benchmark_weights)
w_sector = cp.Variable(n_sectors)

# Constraints: Sector weights must stay within ±5% of the benchmark
sector_constraints = []
sector_weight_values = list(sector_benchmark_weights.values())
for i in range(n_sectors):
    sector_constraints.append(w_sector[i] >= sector_weight_values[i] - 0.05)
    sector_constraints.append(w_sector[i] <= sector_weight_values[i] + 0.05)

# Objective: Maximize expected return of the sector portfolio
sector_expected_returns = {sector: returns[[t for t in filtered_tickers if sectors.get(t) == sector]].mean().mean() * 252
                           for sector in sector_benchmark_weights.keys()}
sector_expected_returns_vector = np.array(list(sector_expected_returns.values()))

portfolio_sector_return = sector_expected_returns_vector @ w_sector

problem = cp.Problem(cp.Maximize(portfolio_sector_return), [cp.sum(w_sector) == 1] + sector_constraints)
problem.solve()

sector_optimal_weights = w_sector.value

# ---------------------------- STEP 4: STOCK-LEVEL OPTIMIZATION (EQUAL RISK CONTRIBUTION) ---------------------------- #
final_stock_weights = {}

for i, (sector, benchmark_weight) in enumerate(sector_benchmark_weights.items()):
    sector_stocks = [ticker for ticker in filtered_tickers if sectors.get(ticker) == sector]
    if len(sector_stocks) < 10:
        print(f"Skipping {sector} due to insufficient stocks")
        continue
    
    # Select 10 stocks randomly for demonstration
    selected_stocks = np.random.choice(sector_stocks, 10, replace=False)
    
    # Calculate stock covariance matrix and risk contributions
    sector_stock_returns = returns[selected_stocks]
    cov_matrix = sector_stock_returns.cov() * 252  # Annualized
    
    # Equal risk contribution optimization
    n = len(selected_stocks)
    w_stock = cp.Variable(n)
    risk_contributions = cp.quad_form(w_stock, cov_matrix)

    constraints = [cp.sum(w_stock) == 1, w_stock >= 0]
    problem = cp.Problem(cp.Minimize(risk_contributions), constraints)
    problem.solve()

    stock_weights = w_stock.value
    for j, stock in enumerate(selected_stocks):
        final_stock_weights[stock] = stock_weights[j] * sector_optimal_weights[i]

# Convert to Pandas Series
portfolio_weights = pd.Series(final_stock_weights)

# ---------------------------- STEP 5: BACKTESTING ---------------------------- #
test_start = "2024-01-01"
test_end = "2024-12-31"
test_data = download_data(list(final_stock_weights.keys()), test_start, test_end)

# Monthly rebalancing
test_returns = test_data.pct_change().dropna()
monthly_returns = test_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

# Compute cumulative portfolio returns with monthly rebalancing
portfolio_cumulative_returns = (1 + monthly_returns @ portfolio_weights).cumprod() - 1
total_return = portfolio_cumulative_returns.iloc[-1]

# Compare Performance
sp500_benchmark = 0.2502  # Given S&P 500 total return in 2024
outperformance = total_return - sp500_benchmark

# ---------------------------- STEP 6: PRINT RESULTS ---------------------------- #
print(f"Total Portfolio Return in 2024: {total_return:.2%}")
print(f"Outperformance vs S&P 500: {outperformance:.2%}")

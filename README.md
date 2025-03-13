# Portfolio Optimization with Two-Layer Strategy

## Overview
This project implements a **two-layer portfolio optimization strategy** for the S&P 500 using sector-level and stock-level optimizations. It scrapes S&P 500 data, applies sector allocation constraints, selects stocks using equal risk contribution, and backtests the portfolio against the S&P 500.

## Features
- **Sector-Level Optimization**: Limits sector weight deviations to ±5% of the benchmark.
- **Stock-Level Optimization**: Uses equal risk contribution to allocate stock weights.
- **Backtesting**: Evaluates the optimized portfolio's performance in 2024 with monthly rebalancing.

## Data Sources
- S&P 500 Constituents: Wikipedia
- Historical stock data: Yahoo Finance (`yfinance` library)

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas yfinance cvxpy
```

## Usage
### 1. Scraping S&P 500 Data
The script extracts S&P 500 constituents and their sector classifications from Wikipedia.

### 2. Downloading Historical Stock Data
It fetches adjusted closing prices (2019-2023) using Yahoo Finance.

### 3. Sector-Level Optimization
- Uses **convex optimization** to adjust sector weights while ensuring they stay within ±5% of the benchmark.
- Maximizes expected portfolio return at the sector level.

### 4. Stock-Level Optimization (Equal Risk Contribution)
- Selects **10 stocks per sector** (randomly chosen for demonstration).
- Allocates stock weights using an **equal risk contribution approach** based on the covariance matrix.

### 5. Backtesting
- Fetches **2024 stock data**.
- Performs **monthly rebalancing**.
- Compares portfolio returns against the **S&P 500 benchmark**.

## Results
- Computes the **total return** for 2024.
- Calculates **outperformance vs. the S&P 500**.

## Output Example
```plaintext
Total Portfolio Return in 2024: 28.75%
Outperformance vs S&P 500: 3.75%
```

## Author
**Princely Oseji**

## License
This project is for educational and research purposes only.


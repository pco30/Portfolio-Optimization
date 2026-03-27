# Portfolio Optimization with Two-Layer Strategy

## Overview
This project implements a two-layer portfolio optimization workflow for the S&P 500:
1) optimize sector allocations under benchmark band constraints, and
2) optimize stock allocations within each sector using a variance-based risk allocation approach.

The project has been refactored into modules to make the pipeline easier to maintain and extend.

## Project Structure
- `src/main.py` - Orchestrates the full workflow from data fetch to reporting.
- `src/config.py` - Centralized constants (dates, benchmark weights, optimization settings).
- `src/data.py` - Data acquisition and preprocessing (Wikipedia scrape + Yahoo Finance download).
- `src/optimization.py` - Sector-level and stock-level optimization routines.
- `src/backtest.py` - Monthly-rebalanced backtest calculations.
- `src/__init__.py` - Marks `src` as a Python package.
- `Portfolio_Optimization.py` - Backward-compatible root entrypoint that calls `src.main.run()`.
- `requirements.txt` - Python dependencies.

## Features
- Sector-level optimization constrained to +/- 5% around benchmark sector weights.
- Stock-level long-only optimization within each sector.
- End-to-end backtest for 2024 with monthly rebalancing.

## Data Sources
- S&P 500 constituents and sectors: Wikipedia.
- Historical stock prices: Yahoo Finance via `yfinance`.

## Installation
Create and activate a virtual environment (recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Run either entrypoint from the project root:

```bash
python -m src.main
```

or

```bash
python Portfolio_Optimization.py
```

Both commands execute the same pipeline:
- scrape S&P 500 constituents,
- download 2019-2023 data for training,
- optimize sector then stock weights,
- backtest on 2024 data,
- print total return and outperformance vs S&P 500.

## Output Example
```text
Total Portfolio Return in 2024: 28.75%
Outperformance vs S&P 500: 3.75%
```

## Notes
- Stock selection per sector is randomly sampled for demonstration.
- Results may differ between runs unless randomness is controlled.

## Author
Princely Oseji

## License
This project is for educational and research purposes only.


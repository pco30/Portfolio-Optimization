"""Portfolio optimization routines."""

from typing import Dict, List

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import MIN_STOCKS_PER_SECTOR, SECTOR_BAND, STOCKS_PER_SECTOR, TRADING_DAYS_PER_YEAR


def optimize_sector_weights(
    returns: pd.DataFrame,
    filtered_tickers: List[str],
    sectors: Dict[str, str],
    sector_benchmark_weights: Dict[str, float],
) -> np.ndarray:
    """Optimize sector weights within benchmark bands."""
    n_sectors = len(sector_benchmark_weights)
    w_sector = cp.Variable(n_sectors)

    sector_constraints = []
    sector_weight_values = list(sector_benchmark_weights.values())
    for i in range(n_sectors):
        sector_constraints.append(w_sector[i] >= sector_weight_values[i] - SECTOR_BAND)
        sector_constraints.append(w_sector[i] <= sector_weight_values[i] + SECTOR_BAND)

    sector_expected_returns = {
        sector: returns[[t for t in filtered_tickers if sectors.get(t) == sector]].mean().mean() * TRADING_DAYS_PER_YEAR
        for sector in sector_benchmark_weights.keys()
    }
    sector_expected_returns_vector = np.array(list(sector_expected_returns.values()))
    portfolio_sector_return = sector_expected_returns_vector @ w_sector

    problem = cp.Problem(cp.Maximize(portfolio_sector_return), [cp.sum(w_sector) == 1] + sector_constraints)
    problem.solve()
    return w_sector.value


def optimize_stock_weights_by_sector(
    returns: pd.DataFrame,
    filtered_tickers: List[str],
    sectors: Dict[str, str],
    sector_benchmark_weights: Dict[str, float],
    sector_optimal_weights: np.ndarray,
) -> Dict[str, float]:
    """Optimize stock allocations within each sector and scale by sector weights."""
    final_stock_weights: Dict[str, float] = {}

    for i, sector in enumerate(sector_benchmark_weights.keys()):
        sector_stocks = [ticker for ticker in filtered_tickers if sectors.get(ticker) == sector]
        if len(sector_stocks) < MIN_STOCKS_PER_SECTOR:
            print(f"Skipping {sector} due to insufficient stocks")
            continue

        selected_stocks = np.random.choice(sector_stocks, STOCKS_PER_SECTOR, replace=False)
        sector_stock_returns = returns[selected_stocks]
        cov_matrix = sector_stock_returns.cov() * TRADING_DAYS_PER_YEAR

        n_stocks = len(selected_stocks)
        w_stock = cp.Variable(n_stocks)
        objective = cp.quad_form(w_stock, cov_matrix)
        constraints = [cp.sum(w_stock) == 1, w_stock >= 0]

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()

        stock_weights = w_stock.value
        for j, stock in enumerate(selected_stocks):
            final_stock_weights[stock] = stock_weights[j] * sector_optimal_weights[i]

    return final_stock_weights

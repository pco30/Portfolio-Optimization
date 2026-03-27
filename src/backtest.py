"""Backtesting helpers."""

from typing import Dict, Tuple

import pandas as pd


def run_monthly_rebalance_backtest(
    test_data: pd.DataFrame,
    final_stock_weights: Dict[str, float],
) -> Tuple[pd.Series, float]:
    """Compute cumulative returns and total return with monthly rebalancing."""
    portfolio_weights = pd.Series(final_stock_weights)
    test_returns = test_data.pct_change().dropna()
    monthly_returns = test_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    portfolio_cumulative_returns = (1 + monthly_returns @ portfolio_weights).cumprod() - 1
    total_return = portfolio_cumulative_returns.iloc[-1]
    return portfolio_cumulative_returns, total_return

"""Main orchestration for portfolio optimization and backtest."""

from .backtest import run_monthly_rebalance_backtest
from .config import (
    SECTOR_BENCHMARK_WEIGHTS,
    SP500_BENCHMARK_2024,
    TEST_END_DATE,
    TEST_START_DATE,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
)
from .data import download_data, get_sp500_tickers_and_sectors, prepare_returns
from .optimization import optimize_sector_weights, optimize_stock_weights_by_sector


def run() -> None:
    """Execute full pipeline from data gathering to backtest reporting."""
    tickers, sectors = get_sp500_tickers_and_sectors()

    training_data = download_data(tickers, TRAIN_START_DATE, TRAIN_END_DATE)
    _, filtered_tickers, returns = prepare_returns(training_data)

    sector_optimal_weights = optimize_sector_weights(
        returns=returns,
        filtered_tickers=filtered_tickers,
        sectors=sectors,
        sector_benchmark_weights=SECTOR_BENCHMARK_WEIGHTS,
    )

    final_stock_weights = optimize_stock_weights_by_sector(
        returns=returns,
        filtered_tickers=filtered_tickers,
        sectors=sectors,
        sector_benchmark_weights=SECTOR_BENCHMARK_WEIGHTS,
        sector_optimal_weights=sector_optimal_weights,
    )

    test_data = download_data(list(final_stock_weights.keys()), TEST_START_DATE, TEST_END_DATE)
    _, total_return = run_monthly_rebalance_backtest(test_data, final_stock_weights)

    outperformance = total_return - SP500_BENCHMARK_2024
    print(f"Total Portfolio Return in 2024: {total_return:.2%}")
    print(f"Outperformance vs S&P 500: {outperformance:.2%}")


if __name__ == "__main__":
    run()

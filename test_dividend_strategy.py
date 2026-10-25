import unittest

import numpy as np
import pandas as pd

from dividend_strategy import (
    StrategyConfig,
    clean_dividends,
    performance,
    select_portfolio,
    transaction_cost,
)


class DividendStrategyTests(unittest.TestCase):
    def test_clean_dividends_deduplicates_plan_updates(self):
        frame = pd.DataFrame(
            {
                "ts_code": ["A", "A", "A"],
                "end_date": pd.to_datetime(["2023-12-31"] * 3),
                "ann_date": pd.to_datetime(["2024-03-01", "2024-04-01", "2024-05-01"]),
                "ex_date": pd.to_datetime(["2024-06-01"] * 3),
                "cash_div_tax": [0.8, 1.0, 1.0],
                "div_proc": ["预案", "实施", "实施"],
            }
        )
        result = clean_dividends(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].cash_div_tax, 1.0)

    def test_selector_prefers_yield_and_low_volatility(self):
        dates = pd.bdate_range("2023-01-01", periods=140)
        raw = pd.DataFrame({"A": 10.0, "B": 10.0, "C": 10.0}, index=dates)
        returns = pd.DataFrame(
            {
                "A": np.tile([0.001, -0.001], 70),
                "B": np.tile([0.01, -0.01], 70),
                "C": np.tile([0.002, -0.002], 70),
            }, index=dates,
        )
        rows = []
        for code, cash in (("A", 1.0), ("B", 0.5), ("C", 0.2)):
            for year in (2022, 2023):
                rows.append({"ts_code": code, "ex_date": pd.Timestamp(year, 6, 1), "cash_div_tax": cash})
        dividends = pd.DataFrame(rows)
        weights, factors = select_portfolio(
            dates[-1], raw, returns, dividends, StrategyConfig(top_n=2)
        )
        self.assertIn("A", weights.index)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertEqual(len(factors), 2)

    def test_performance_detects_drawdown(self):
        stats = performance(pd.Series([0.0, 0.10, -0.20, 0.05]))
        self.assertLess(stats["max_drawdown"], 0)

    def test_cost_is_positive_and_stamp_is_sell_only_approximation(self):
        cost = transaction_cost(0.5, 1_000_000, 10, StrategyConfig())
        self.assertGreater(cost, 0)
        self.assertLess(cost, 0.01)


if __name__ == "__main__":
    unittest.main()

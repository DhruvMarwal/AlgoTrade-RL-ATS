"""Small deterministic tests for the trading environment's accounting."""
import unittest
import numpy as np
import pandas as pd

from env.trading_env import TradingEnv, BUY, HOLD, SELL


class TradingEnvAccountingTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2024-01-01", periods=4, freq="D")
        self.features = pd.DataFrame({"x": [0.0, 0.0, 0.0, 0.0]}, index=self.dates)
        self.returns = np.array([0.0, np.log(1.10), np.log(0.90), np.log(1.20)])

    def test_buy_hold_sell_costs_and_next_day_timing(self):
        env = TradingEnv(self.features, self.returns, transaction_cost=0.01)
        env.reset()
        env.step(BUY)   # earns r[1] (the next day's return)
        env.step(HOLD)  # earns r[2]
        _, _, done, _, _ = env.step(SELL)  # exits and pays cost; no r[3] while flat
        self.assertTrue(done)
        frame = env.to_frame()
        expected = (1 - 0.01) * 1.10 * 0.90 * (1 - 0.01)
        self.assertAlmostEqual(float(frame["equity"].iloc[-1]), expected, places=8)
        self.assertEqual(frame["traded"].tolist(), [1, 0, 1])
        self.assertEqual(frame["position"].tolist(), [1, 1, 0])
        self.assertEqual(frame["pnl_date"].iloc[0], self.dates[1])

    def test_repeated_buy_is_not_a_trade(self):
        env = TradingEnv(self.features, self.returns, transaction_cost=0.01)
        env.reset()
        env.step(BUY)
        env.step(BUY)
        frame = env.to_frame()
        self.assertEqual(frame["traded"].tolist(), [1, 0])
        self.assertEqual(frame["executed_action"].tolist(), [BUY, BUY])

    def test_turnover_penalty_does_not_change_reported_pnl(self):
        env = TradingEnv(self.features, self.returns, transaction_cost=0.01, turnover_penalty=0.001)
        env.reset()
        _, reward, _, _, _ = env.step(BUY)
        frame = env.to_frame()
        expected_net = np.log1p(-0.01) + self.returns[1]
        self.assertAlmostEqual(float(frame["strategy_logret"].iloc[0]), expected_net, places=8)
        self.assertAlmostEqual(float(reward), (expected_net - 0.001) * 100.0, places=8)


if __name__ == "__main__":
    unittest.main()

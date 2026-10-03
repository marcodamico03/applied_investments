"""
Operating leverage: operating costs relative to assets. Ingredient of slow_combo
(on its own it is a known anomaly).

Source: Novy-Marx (2011) "Operating leverage". Firms with high operating costs
(largely fixed in the short run) have riskier, more cyclical profits and earn
higher average returns.

For formation year t (held July t - June t+1), fiscal year ending in t-1:
  signal = (cogs + xsga) / at        (missing xsga counts as 0)
"""
import pandas as pd

from src.signals._yearly import to_rows, yearly

COMPUSTAT_COLS = ["cogs", "xsga", "at"]
REBALANCE = "annual"
LONG, SHORT = 10, 1


def op_leverage_yearly(panel: pd.DataFrame) -> pd.Series:
    acc = yearly(panel, ["cogs", "xsga", "at"])
    return (acc["cogs"] + acc["xsga"].fillna(0)) / acc["at"].where(acc["at"] > 0)


def op_leverage(panel: pd.DataFrame) -> pd.Series:
    return to_rows(panel, op_leverage_yearly(panel))

"""
Performance of a long-short spread, before and after trading costs.

t-stat:  mean / (std / sqrt(months)), as in the slides (no Newey-West here;
         the alphas in alpha.py have Newey-West t-stats).
Sharpe:  annualised, monthly mean / monthly std * sqrt(12). A long-short spread
         costs nothing to set up, so no risk-free rate is subtracted.
"""
from typing import Dict

import numpy as np
import pandas as pd


def _mean_t_sharpe(r: pd.Series):
    r = r.dropna()
    mean, std = r.mean(), r.std()
    return mean, mean / (std / np.sqrt(len(r))), mean / std * np.sqrt(12), len(r)


def spread_stats(
    gross: pd.Series, net: pd.Series, turnover_long: pd.Series, turnover_short: pd.Series
) -> Dict[str, float]:
    """Average (% per month), t-stat and Sharpe ratio of the spread before and
    after costs, and the average monthly one-way turnover (%) of each leg."""
    g_mean, g_t, g_sharpe, months = _mean_t_sharpe(gross)
    n_mean, n_t, n_sharpe, _ = _mean_t_sharpe(net)
    return {
        "Avg %": g_mean * 100,
        "t": g_t,
        "Sharpe": g_sharpe,
        "Net avg %": n_mean * 100,
        "Net t": n_t,
        "Net Sharpe": n_sharpe,
        "Turnover % long": turnover_long.mean() * 100,
        "Turnover % short": turnover_short.mean() * 100,
        "Months": months,
    }

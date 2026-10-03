"""
Return seasonality (benchmark version, on raw returns): stocks that did well in
the same calendar month in past years tend to do well in that month again.

Source: Heston and Sadka (2008) "Seasonality in the cross-section of stock
returns"; the "same month minus other months" form is from Keloharju,
Linnainmaa and Nyberg (2016) "Return seasonalities".
This raw version is the benchmark for seasonality_idio.py (our version).

In the row of month t (portfolio held in month t+1):
  same  = average return in the calendar month of t+1, in each of the last
          YEARS years (months t-11, t-23, ..., t-12*YEARS+1)
  other = average return in all the other months of the same YEARS years
  signal = same - other     (at least MIN_YEARS same-month observations)
"""
import numpy as np
import pandas as pd

from src.signals._common import from_matrix, to_matrix

COMPUSTAT_COLS = []
REBALANCE = "monthly"
LONG, SHORT = 10, 1
YEARS, MIN_YEARS = 20, 5


def seasonal_spread(M: np.ndarray, years: int = YEARS, min_years: int = MIN_YEARS) -> np.ndarray:
    """Same-calendar-month average minus other-months average over the last `years`
    years, for every row t of a (months x stocks) matrix. Uses rows t-12*years+1..t only."""
    T = M.shape[0]
    filled = np.nan_to_num(M)
    has = (~np.isnan(M)).astype(float)
    # running sums over the last 12*years months (rows t-12*years+1 .. t)
    cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(filled, axis=0)])
    cn = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(has, axis=0)])
    L = 12 * years
    out = np.full(M.shape, np.nan)
    for t in range(11, T):
        lo = max(0, t - L + 1)
        tot_s, tot_n = cs[t + 1] - cs[lo], cn[t + 1] - cn[lo]
        lags = np.arange(t - 11, lo - 1, -12)          # t-11, t-23, ... (same month as t+1)
        same_s, same_n = filled[lags].sum(axis=0), has[lags].sum(axis=0)
        other_n = tot_n - same_n
        ok = (same_n >= min_years) & (other_n >= 11 * min_years)
        out[t, ok] = same_s[ok] / same_n[ok] - (tot_s[ok] - same_s[ok]) / other_n[ok]
    return out


def seasonality(panel: pd.DataFrame) -> pd.Series:
    M, months, permnos = to_matrix(panel, "MthRet")
    return from_matrix(panel, seasonal_spread(M), months, permnos)

"""
Hybrid: half slow_combo (annual, low turnover), half idiosyncratic seasonality
(monthly, strong alpha but high turnover). Rebalanced monthly.

Idea: the seasonality signal changes every month (it looks at a different
calendar month), so on its own its portfolios are replaced almost entirely each
month and trading costs eat the alpha. The slow half of this signal changes only
once a year, so a stock's combined rank moves less from month to month: fewer
trades, while keeping the seasonality alpha. Combining signals to net out trades
is the point of DeMiguel, Martin-Utrera, Nogales and Uppal (2020) "A
transaction-cost perspective on the multitude of firm characteristics".

Every month:
  slow part     = rank of slow_combo (already neutralised, see slow_combo.py)
  seasonal part = seasonality_idio neutralised against size, book-to-market,
                  profitability, asset growth, momentum and last-month return
                  (same method as alpha_combo2.py), ranked again
  signal        = WEIGHT_SLOW x slow part + (1 - WEIGHT_SLOW) x seasonal part
Both parts are required.
"""
import pandas as pd

from src.signals._common import cross_rank
from src.signals.alpha_combo2 import controls, neutralise
from src.signals.seasonality_idio import seasonality_idio
from src.signals.slow_combo import slow_combo

COMPUSTAT_COLS = ["be", "at", "at_lag", "sale", "cogs", "xsga", "xrd"]
CRSP_COLS = ["me_firm"]
REBALANCE = "monthly"
LONG, SHORT = 10, 1
WEIGHT_SLOW = 0.5


def hybrid_combo(panel: pd.DataFrame) -> pd.Series:
    print("  hybrid_combo: slow part...", flush=True)
    slow = cross_rank(panel, slow_combo(panel))
    print("  hybrid_combo: seasonal part...", flush=True)
    ctrl_ranks = controls(panel).apply(lambda col: cross_rank(panel, col))
    seasonal = neutralise(panel, seasonality_idio(panel), ctrl_ranks)
    return WEIGHT_SLOW * slow + (1 - WEIGHT_SLOW) * seasonal

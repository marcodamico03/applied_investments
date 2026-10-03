"""
Idiosyncratic return seasonality (our version): the seasonality signal of
seasonality.py computed on FF5 residual returns instead of raw returns.

Why: Keloharju, Linnainmaa and Nyberg (2016) argue that much of the stock-level
seasonality comes from seasonality in the factors (January, size, ...). Removing
the factor part first keeps only the stock's own calendar pattern, which no
factor model can price. If the signal still works, that is new evidence; if it
does not, it supports their explanation. Either result is reportable.

Residual of month s: excess return minus alpha + FF5 betas x factors, betas from
months s-35..s (src/signals/_common.py). Same/other months as in seasonality.py.
"""
import pandas as pd

from src.signals._common import factor_residuals, from_matrix
from src.signals.seasonality import seasonal_spread

COMPUSTAT_COLS = []
REBALANCE = "monthly"
LONG, SHORT = 10, 1
MODEL, WINDOW = "FF5", 36


def seasonality_idio(panel: pd.DataFrame) -> pd.Series:
    R, months, permnos = factor_residuals(panel, MODEL, WINDOW)
    return from_matrix(panel, seasonal_spread(R.astype(float)), months, permnos)

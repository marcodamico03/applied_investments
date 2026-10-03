"""
R&D intensity: R&D capital relative to market value. Ingredient of slow_combo
(on its own it is a known anomaly).

Source: Chan, Lakonishok and Sougiannis (2001) "The stock market valuation of
research and development expenditures"; Lev and Sougiannis (1996). Accounting
expenses R&D immediately, so R&D-intensive firms look expensive on book value
and the market undervalues their future profits.

For formation year t (held July t - June t+1):
  R&D capital = xrd_t + 0.8 xrd_t-1 + 0.6 xrd_t-2 + 0.4 xrd_t-3 + 0.2 xrd_t-4
                (fiscal years ending in t-1, t-2, ...; a missing year counts as 0)
  signal      = R&D capital / firm market cap at the end of December t-1
Only firms with positive R&D in the latest fiscal year.
"""
import pandas as pd

from src.signals._yearly import december, lag_years, to_rows, yearly

COMPUSTAT_COLS = ["xrd"]
CRSP_COLS = ["me_firm"]
REBALANCE = "annual"
LONG, SHORT = 10, 1


def rd_capital_to_me(panel: pd.DataFrame) -> pd.Series:
    """One value per (PERMNO, formation_year)."""
    xrd = yearly(panel, ["xrd"])["xrd"]
    cap = xrd.fillna(0).copy()
    for k, w in zip(range(1, 5), (0.8, 0.6, 0.4, 0.2)):
        cap = cap + w * lag_years(xrd, k).reindex(cap.index).fillna(0)
    me = december(panel, "me_firm") / 1000      # CRSP thousands -> millions
    me = me[~me.index.duplicated()].reindex(cap.index)
    return (cap / me).where((xrd > 0) & (me > 0))


def rd_intensity(panel: pd.DataFrame) -> pd.Series:
    return to_rows(panel, rd_capital_to_me(panel))

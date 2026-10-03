"""
Composite equity issuance over 5 years (sign flipped: high = repurchases).
Ingredient of slow_combo (on its own it is a known anomaly).

Source: Daniel and Titman (2006) "Market reactions to tangible and intangible
information". Firms that issue shares (equity offerings, stock-financed deals,
employee stock) earn low returns; firms that buy back shares or pay out cash earn
high returns. Uses CRSP only.

For formation year t (held July t - June t+1), with December t-1 = month 0:
  ISS = log(ME_0 / ME_-60) - sum of log(1 + monthly return) over months -59..0
        (growth in market value not explained by the stock's return = net
        shares issued; dividends count as payout, as in Daniel and Titman)
  signal = -ISS                       (long repurchasers, short issuers)
All 60 monthly returns are required. ME = market cap of the share class (PERMNO).
"""
import numpy as np
import pandas as pd

from src.signals._common import from_matrix, to_matrix
from src.signals._yearly import december, to_rows

COMPUSTAT_COLS = []
REBALANCE = "annual"
LONG, SHORT = 10, 1
MONTHS = 60


def issuance_yearly(panel: pd.DataFrame) -> pd.Series:
    """-ISS in December of t-1, for formation year t."""
    R, months, permnos = to_matrix(panel, "MthRet")
    ME, _, _ = to_matrix(panel, "mkt_cap")
    logr = np.log1p(np.clip(R, -0.99, None))
    cum = pd.DataFrame(logr).rolling(MONTHS, min_periods=MONTHS).sum().to_numpy()
    me_then = np.vstack([np.full((MONTHS, ME.shape[1]), np.nan), ME[:-MONTHS]])
    with np.errstate(divide="ignore", invalid="ignore"):
        iss = np.log(ME / me_then) - cum
    neg_iss = from_matrix(panel, -iss, months, permnos)
    return december(panel.assign(neg_iss=neg_iss), "neg_iss")


def comp_issuance(panel: pd.DataFrame) -> pd.Series:
    return to_rows(panel, issuance_yearly(panel))

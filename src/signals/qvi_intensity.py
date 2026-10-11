"""
Idea 3: qvi_intan_full plus a third piece, intangible intensity: R&D / ME and
advertising / ME, ranked within the industry (one third of the score).
Why: in Hou, Xue and Zhang (2020) the intangible-investment-to-price signals are
among the few the q-factor model cannot price: markets undervalue firms that invest
in invisible capital, compared with peers in the same industry.
"""
import pandas as pd

from src.signals._qvi_plus import COMPUSTAT_COLS, CRSP_COLS, qvi_plus_year  # noqa: F401
from src.signals._yearly import to_rows

REBALANCE = "annual"
LONG, SHORT = 10, 1


def qvi_intensity(panel: pd.DataFrame) -> pd.Series:
    return to_rows(panel, qvi_plus_year(panel, intensity=True))

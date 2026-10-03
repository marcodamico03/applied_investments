"""
Slow combo: a low-turnover composite of three slow signals, each first
neutralised against the characteristics the benchmark factors are built on.
Rebalanced once a year (end of June), so trading costs are low.

Ingredients (formulas in their own files):
  rd_intensity   R&D capital / market cap         (Chan, Lakonishok, Sougiannis 2001)
  op_leverage    (cogs + xsga) / assets            (Novy-Marx 2011)
  comp_issuance  minus 5-year composite issuance   (Daniel and Titman 2006)

Why combine and neutralise: on its own each signal is partly value,
profitability or investment, which FF5/FF6 and q5 already price. Each formation
year, every signal's rank is regressed on the ranks of size, book-to-market,
operating profitability, asset growth and momentum 12-2 (all known at the end of
June t), and only the residual is kept. The three residuals are ranked again and
averaged (at least MIN_SIGNALS available; R&D exists only for R&D firms). What is
left is the part of each signal that the benchmark factors cannot explain.
Inspiration for the method: Lewellen (2015); Stambaugh and Yuan (2017).
"""
import pandas as pd

from src.signals._yearly import annual_controls, neutralise_yearly, to_rows, year_rank
from src.signals.comp_issuance import issuance_yearly
from src.signals.op_leverage import op_leverage_yearly
from src.signals.rd_intensity import rd_capital_to_me

COMPUSTAT_COLS = ["be", "at", "at_lag", "sale", "cogs", "xsga", "xrd"]
CRSP_COLS = ["me_firm"]
REBALANCE = "annual"
LONG, SHORT = 10, 1
MIN_SIGNALS = 2


def slow_combo_yearly(panel: pd.DataFrame) -> pd.Series:
    """One value per (PERMNO, formation_year)."""
    ctrl = annual_controls(panel)
    parts = {}
    for name, fn in [("rd_intensity", rd_capital_to_me),
                     ("op_leverage", op_leverage_yearly),
                     ("comp_issuance", issuance_yearly)]:
        print(f"  slow_combo: {name}...", flush=True)
        s = fn(panel)
        parts[name] = neutralise_yearly(s[~s.index.duplicated()], ctrl)
    parts = pd.DataFrame(parts)
    combo = parts.mean(axis=1).where(parts.notna().sum(axis=1) >= MIN_SIGNALS)
    return year_rank(combo.dropna())


def slow_combo(panel: pd.DataFrame) -> pd.Series:
    return to_rows(panel, slow_combo_yearly(panel))

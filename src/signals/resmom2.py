"""
Residual momentum 2.0: momentum in the part of a stock's return that the
Fama-French 5 factors do not explain, made "purer" in three ways.

Inspiration: Blitz, Huij and Martens (2011) "Residual momentum"; Blitz, Hanauer
and Vidojevic (2020) "The idiosyncratic momentum anomaly"; Novy-Marx (2012) "Is
momentum really momentum?" (the intermediate 12-7 window); Da, Gurun and
Warachka (2014) "Frog in the pan" (steady drifts beat jumps); Moskowitz and
Grinblatt (1999) "Do industries explain momentum?".

Residual of month s = excess return minus alpha + FF5 betas x factors, with the
betas estimated on months s-35..s (src/signals/_common.py).

In the row of month t (portfolio formed at the end of t, held in t+1), using the
residuals of months t-11..t-1 (the most recent month t is skipped, as in
standard momentum), all 11 required:
  a) standardised residual momentum  = sum of the 11 residuals / their std
  b) intermediate residual momentum  = sum of residuals t-11..t-6 / same std
  c) consistency (no jumps)          = (% positive - % negative) residual months
score = average of the monthly ranks of a, b and c, minus the average score of
the stock's industry (SIC 2 digits, industries with at least 5 stocks), so the
signal does not pick whole industries (that part is in the momentum factor).
"""
import numpy as np
import pandas as pd

from src.signals._common import cross_rank, factor_residuals, from_matrix

COMPUSTAT_COLS = []
REBALANCE = "monthly"
LONG, SHORT = 10, 1
MODEL, WINDOW = "FF5", 36


def resmom2(panel: pd.DataFrame) -> pd.Series:
    R, months, permnos = factor_residuals(panel, MODEL, WINDOW)
    T = len(months)
    a = np.full_like(R, np.nan)
    b = np.full_like(R, np.nan)
    c = np.full_like(R, np.nan)
    for t in range(12, T):
        win = R[t - 11: t]              # months t-11 .. t-1
        ok = ~np.isnan(win).any(axis=0)
        if not ok.any():
            continue
        w = win[:, ok].astype(float)
        sd = w.std(axis=0, ddof=1)
        sd[sd == 0] = np.nan
        a[t, ok] = w.sum(axis=0) / sd
        b[t, ok] = w[:6].sum(axis=0) / sd   # months t-11 .. t-6
        c[t, ok] = ((w > 0).sum(axis=0) - (w < 0).sum(axis=0)) / 11

    score = sum(cross_rank(panel, from_matrix(panel, M, months, permnos)) for M in (a, b, c)) / 3

    # Industry-neutral: subtract the industry average of the score in that month
    sic2 = (pd.to_numeric(panel["SICCD"], errors="coerce") // 100)
    grp = [panel["MthCalDt"], sic2]
    ind_mean = score.groupby(grp).transform("mean")
    ind_n = score.groupby(grp).transform("count")
    return (score - ind_mean.where(ind_n >= 5, 0.0)).where(score.notna())

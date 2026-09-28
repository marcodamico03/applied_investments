"""
Univariate portfolio sorts, following the method of the Applied Investments slides.

How it works, in plain words:
1. At each formation date, keep the stocks that have a valid signal and a
   positive market cap on that date.
2. Compute breakpoints (e.g. deciles) using NYSE stocks only.
3. Put every stock (NYSE, AMEX, NASDAQ) into a portfolio: stocks with a signal
   <= the first breakpoint go in portfolio 1, and so on.
4. Hold the portfolios until the next formation date and compute monthly
   equal-weighted and value-weighted returns. Value weights use the market cap
   at the end of the previous month (for July: the June market cap).

Two rebalancing modes:
- "annual": sort at the end of June of year t, hold July t to June t+1.
  Use this for accounting signals. The signal must be constant within each
  July-June year (the check below enforces it, to avoid look-ahead bias).
- "monthly": sort at the end of every month using that month's signal, hold
  the next month. Use this for market signals (momentum, volatility, ...).
  The signal in a row of month m must be known at the end of month m.
"""
from typing import Dict, Sequence

import numpy as np
import pandas as pd

REQUIRED_COLS = ["PERMNO", "MthCalDt", "formation_year", "PrimaryExch", "MthRet", "mkt_cap"]


def prepare_panel(panel: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Keep the needed columns, sort, and add the lagged market cap (weights)."""
    missing = [c for c in REQUIRED_COLS + [signal] if c not in panel.columns]
    if missing:
        raise KeyError(f"Columns missing from the panel: {missing}")

    df = panel[REQUIRED_COLS + [signal]].copy()
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    df = df.sort_values(["PERMNO", "MthCalDt"]).reset_index(drop=True)

    # A simple month counter (Jan 1963 -> 23556, Feb 1963 -> 23557, ...)
    df["month_id"] = df["MthCalDt"].dt.year * 12 + df["MthCalDt"].dt.month - 1

    # Market cap at the end of the PREVIOUS month, only if that month exists
    prev_month = df.groupby("PERMNO")["month_id"].shift(1)
    prev_mcap = df.groupby("PERMNO")["mkt_cap"].shift(1)
    df["mcap_lag"] = np.where(df["month_id"] - prev_month == 1, prev_mcap, np.nan)
    return df


def _formation_snapshot(df: pd.DataFrame, signal: str, rebalance: str) -> pd.DataFrame:
    """One row per stock and formation date, with the value used for sorting.
    Also sets df["form_key"]: which formation date each holding month belongs to."""
    if rebalance == "annual":
        # Look-ahead protection: the signal must not change within July-June
        n_values = (
            df.dropna(subset=[signal]).groupby(["PERMNO", "formation_year"])[signal].nunique()
        )
        if (n_values > 1).any():
            raise ValueError(
                f"'{signal}' changes within a July-June year for {int((n_values > 1).sum())} "
                "stock-years. Annual rebalancing needs a signal fixed within the year "
                "(accounting data). Use rebalance='monthly' for market-based signals."
            )
        # Market cap and exchange measured at the end of June of year t
        snap = df.loc[df["MthCalDt"].dt.month == 6, ["PERMNO", "MthCalDt", "PrimaryExch", "mkt_cap"]].copy()
        snap["form_key"] = snap["MthCalDt"].dt.year
        # Signal used for the sort in June t = accounting value for July t to June t+1
        sig = (
            df.dropna(subset=[signal])
            .groupby(["PERMNO", "formation_year"])[signal].first()
            .rename("sort_value").reset_index()
            .rename(columns={"formation_year": "form_key"})
        )
        snap = snap.merge(sig, on=["PERMNO", "form_key"], how="inner")
        df["form_key"] = df["formation_year"]

    elif rebalance == "monthly":
        snap = df[["PERMNO", "MthCalDt", "PrimaryExch", "mkt_cap", signal, "month_id"]].copy()
        snap = snap.rename(columns={signal: "sort_value", "month_id": "form_key"})
        df["form_key"] = df["month_id"] - 1  # hold in month m what was formed in m-1

    else:
        raise ValueError("rebalance must be 'annual' or 'monthly'")

    # Eligible at formation: positive market cap and a real (finite) signal
    ok = (snap["mkt_cap"] > 0) & np.isfinite(snap["sort_value"])
    return snap[ok].copy()


def sort_portfolios(
    panel: pd.DataFrame,
    signal: str,
    n_portfolios: int = 10,
    rebalance: str = "annual",
    breakpoint_exchanges: Sequence[str] = ("N",),
) -> Dict[str, pd.DataFrame]:
    """Build portfolios sorted on `signal`.

    Returns a dictionary with:
      ew          monthly equal-weighted returns (rows: months, columns: portfolios)
      vw          monthly value-weighted returns
      n_stocks    number of stocks with a return, per month and portfolio
      breakpoints breakpoints used at each formation date
      assignments which portfolio each stock was in at each formation date
    """
    df = prepare_panel(panel, signal)
    snap = _formation_snapshot(df, signal, rebalance)

    # Breakpoints from NYSE stocks only
    quantiles = [i / n_portfolios for i in range(1, n_portfolios)]
    bp_universe = snap[snap["PrimaryExch"].isin(breakpoint_exchanges)]
    breakpoints = bp_universe.groupby("form_key")["sort_value"].quantile(quantiles).unstack()
    breakpoints.columns = [f"P{round(q * 100)}" for q in quantiles]

    # Assign portfolios: portfolio = 1 + number of breakpoints strictly below the value
    # (so a value equal to a breakpoint goes to the lower portfolio, as on slide 15)
    snap = snap[snap["form_key"].isin(breakpoints.index)]
    bp_matrix = breakpoints.loc[snap["form_key"]].to_numpy()
    values = snap["sort_value"].to_numpy()[:, None]
    snap["port"] = (values > bp_matrix).sum(axis=1) + 1

    # Holding months: attach each stock-month to its portfolio
    hold = df.merge(snap[["PERMNO", "form_key", "port"]], on=["PERMNO", "form_key"], how="inner")
    hold = hold[hold["MthRet"].notna()]
    hold["month"] = hold["MthCalDt"].dt.to_period("M")

    ew = hold.groupby(["month", "port"])["MthRet"].mean().unstack()

    hv = hold[hold["mcap_lag"] > 0].copy()
    hv["w_ret"] = hv["MthRet"] * hv["mcap_lag"]
    grouped = hv.groupby(["month", "port"])
    vw = (grouped["w_ret"].sum() / grouped["mcap_lag"].sum()).unstack()

    n_stocks = hold.groupby(["month", "port"]).size().unstack()

    return {
        "ew": ew,
        "vw": vw,
        "n_stocks": n_stocks,
        "breakpoints": breakpoints,
        "assignments": snap[["PERMNO", "form_key", "sort_value", "port", "mkt_cap"]],
    }


def summarize(returns: pd.DataFrame, long_port: int, short_port: int) -> pd.DataFrame:
    """Average, standard deviation (both in % per month), t-stat and number of
    months for each portfolio and for the long-short spread."""
    r = returns.copy()
    spread = f"{long_port}-{short_port}"
    r[spread] = r[long_port] - r[short_port]
    n = r.count()
    mean = r.mean()
    std = r.std()
    tstat = mean / (std / np.sqrt(n))
    out = pd.DataFrame(
        {"Average %": mean * 100, "StdDev %": std * 100, "t-stat": tstat, "Months": n}
    ).T
    return out

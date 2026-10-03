"""
Portfolio turnover and trading costs.

Turnover of a portfolio in month t, in plain words: the fraction of the portfolio
you have to sell (and buy back with other stocks) at the start of month t.
    turnover(t) = 1/2 * sum over stocks of |w_start(t) - w_end(t-1)|
w_start(t) is the weight the portfolio wants at the start of month t, w_end(t-1)
is last month's weight after one month of returns (weights drift with prices).
0 = nothing traded, 1 = every stock replaced.

Equal-weighted portfolios go back to equal weights every month, so they trade a
little every month even when the list of stocks does not change. Value-weighted
portfolios mostly trade at the formation date (plus delistings and share issues).

Trading cost: every dollar bought or sold pays `one_way_bps`. A turnover of x
means x sold plus x bought, so the cost of one portfolio is 2 * x * one_way_bps.
"""
import pandas as pd


def turnover(holdings: pd.DataFrame, weighting: str) -> pd.DataFrame:
    """Monthly one-way turnover (rows: months, columns: portfolios).
    holdings: the 'holdings' table of sort_portfolios(..., return_holdings=True).
    The weights are the ones sort_portfolios uses for the returns. The first month
    of a portfolio (nothing held the month before) is left empty."""
    if weighting == "vw":
        h = holdings[holdings["mcap_lag"] > 0]
        size = h["mcap_lag"]
    elif weighting == "ew":
        h = holdings
        size = pd.Series(1.0, index=h.index)
    else:
        raise ValueError("weighting must be 'ew' or 'vw'")

    group = [h["month"], h["port"]]
    w_start = size / size.groupby(group).transform("sum")
    grown = w_start * (1 + h["MthRet"])
    w_end = grown / grown.groupby(group).transform("sum")

    # Compare the start weights of month t with the end weights of month t-1
    keys = ["PERMNO", "port", "month"]
    now = h[keys].assign(w_start=w_start)
    before = h[keys].assign(month=h["month"] + 1, w_end_prev=w_end)
    trades = now.merge(before, on=keys, how="outer").fillna({"w_start": 0.0, "w_end_prev": 0.0})
    trades["change"] = (trades["w_start"] - trades["w_end_prev"]).abs()
    result = 0.5 * trades.groupby(["month", "port"])["change"].sum().unstack()

    # Only months where the portfolio is held this month AND was held last month
    held = h.groupby(["month", "port"]).size().unstack()
    held_before = held.reindex(held.index - 1).set_axis(held.index)
    return result.reindex(index=held.index, columns=held.columns).where(
        held.notna() & held_before.notna()
    )


def trading_cost(turnover_long: pd.Series, turnover_short: pd.Series, one_way_bps: float) -> pd.Series:
    """Monthly cost (as a return) of a long-short spread: both legs pay.
    A month with unknown turnover (the very first one) costs nothing."""
    per_dollar = one_way_bps / 10_000
    return 2 * per_dollar * (turnover_long.fillna(0) + turnover_short.fillna(0))

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

Trading cost: buying or selling a stock costs half its bid-ask spread, relative to
the midpoint, (ask - bid) / (ask + bid). The return of month t starts from the
close of the last trading day of month t-1, so that close is when the backtest
trades, and every trade pays the stock's closing spread of that day
(data/interim/spreads_clean.parquet, written by src/data/clean.py):
    cost(t) = sum over stocks of |w_start(t) - w_end(t-1)| * half_spread(end of t-1)
A stock with no quote is charged the average spread of the portfolio's other trades
that month. A month without any quote (before 1993, see clean.py) has an unknown
cost: NaN, so the net spread starts in 1993-01.
"""
import pandas as pd


def weight_changes(holdings: pd.DataFrame, weighting: str) -> pd.DataFrame:
    """One row per stock, portfolio and month t: change = |w_start(t) - w_end(t-1)|.
    holdings: the 'holdings' table of sort_portfolios(..., return_holdings=True).
    The weights are the ones sort_portfolios uses for the returns. Only months where
    the portfolio is held this month AND was held last month (the first month of a
    portfolio has nothing to compare with)."""
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

    held = h[["month", "port"]].drop_duplicates()
    held_both = held.merge(held.assign(month=held["month"] + 1), on=["month", "port"])
    return trades.merge(held_both, on=["month", "port"])[keys + ["change"]]


def turnover(changes: pd.DataFrame) -> pd.DataFrame:
    """Monthly one-way turnover (rows: months, columns: portfolios).
    changes: weight_changes(holdings, weighting)."""
    return 0.5 * changes.groupby(["month", "port"])["change"].sum().unstack()


def trading_cost(changes: pd.DataFrame, spreads: pd.DataFrame) -> pd.DataFrame:
    """Monthly trading cost of each portfolio, as a return (rows: months, columns:
    portfolios). changes: weight_changes(holdings, weighting); spreads: load_spreads()
    of src/data/panel.py."""
    # the trades of month t are made at the last close of month t-1
    quotes = spreads.assign(month=spreads["month"] + 1)
    t = changes.merge(quotes, on=["PERMNO", "month"], how="left")
    group = [t["month"], t["port"]]
    paid = (t["change"] * t["half_spread"]).groupby(group).sum()
    traded = t["change"].groupby(group).sum()
    traded_quoted = t["change"].where(t["half_spread"].notna(), 0.0).groupby(group).sum()
    # trades without a quote pay the average of the quoted ones; none quoted: 0/0 = NaN
    cost = paid / traded_quoted * traded
    return cost.where(traded > 0, 0.0).unstack()

# Cheap and cash-profitable, within each industry

Branch `roberto`. Signal file: `src/signals/quality_value_industry.py`.
Everything below is reproduced by `python scripts/run_roberto.py` (section 7).

## 1. The idea in one sentence

Buy the US stocks that are cheap **for their industry** and generate real **cash** from their
operations; short the stocks that are expensive for their industry and do not.

## 2. The rules

Once a year, at the end of June of year t, using the accounts of the fiscal year that ended
in calendar year t-1 (public by then) and the market value of December t-1:

1. **Universe:** US common stocks (NYSE, AMEX, NASDAQ) with a positive market value in June.
2. **Industry:** each stock gets one of the 12 Fama-French industries from its SIC code.
3. **Cheapness score.** Inside each industry, rank every firm on four ratios and average the
   ranks (at least three needed):
   - book value / market value
   - earnings / market value
   - cash flow (earnings + depreciation) / market value
   - EBITDA / enterprise value

   Banks, insurers and other financials (SIC 6000-6999) use only the first two, because
   for them debt is raw material and EBITDA has no meaning.
4. **Quality score.** Inside each industry, rank every firm on cash-based operating
   profitability (Ball, Gerakos, Linnainmaa and Nikolaev 2016): sales - cost of goods -
   SG&A + R&D, minus the increase in receivables and inventory, plus the increase in
   payables, divided by total assets.
5. **Final score:** the average of the two ranks, again ranked inside the industry.
   Industries need at least 10 stocks that year.
6. **Portfolio:** deciles with NYSE breakpoints. Long decile 10 (cheap and profitable),
   short decile 1 (expensive and unprofitable), each stock weighted by its market value.
7. **Hold for 12 months**, then repeat.

Trading costs follow the team's model (`src/portfolio/costs.py`): half the bid-ask spread
plus price impact, for a fund of $100m per side in 2025 dollars.

## 3. Why it should work

- **Industry-relative cheapness removes a bet that does not pay.** A bank always looks
  cheap next to a software firm. A plain value portfolio is therefore mostly long banks and
  utilities and short technology: an industry bet that adds risk but little return. Inside
  an industry, cheapness is real information (Asness, Porter and Stevens 2000; Cohen and
  Polk 1998).
- **Cash, not accounting profit.** Profit that never arrives as cash is often an accounting
  choice, and investors overpay for it (Sloan 1996; Ball et al. 2016).
- **Each half protects the other.** Cheap stocks alone are often cheap for a reason (value
  traps); profitable stocks alone are often expensive. Requiring both keeps the cheap firms
  that are healthy (Novy-Marx 2013; Asness, Frazzini and Pedersen 2019).
- **Likely reason the premium exists:** investors pay too much for exciting stories and too
  little for boring, cheap, cash-rich firms. Prices correct slowly, and fastest when stories
  collapse, which is why the strategy earns most in crises.

## 4. Results, 1963-07 to 2015-12, value-weighted, after trading costs

Training 1963-2005, validation 2006-2015. The holdout 2016-2025 is reported in section 6.

### The long-short spread

| | Per month | t-stat |
|---|---|---|
| Gross return | 0.63% | 4.84 |
| Trading costs | 0.12% | |
| **Net return** | **0.51%** | **3.90** |
| Net, training | 0.49% | 3.35 |
| Net, validation | 0.59% | 2.03 |

Net Sharpe 0.54. Turnover about 5% per month per side; the strategy breaks even at a cost of
about 300 bps per trade.

### Alphas of the net spread (Newey-West t, 6 lags)

| Model | Net alpha per month | t |
|---|---|---|
| CAPM | 0.62% | 4.67 |
| Fama-French 5 | 0.43% | 3.33 |
| Fama-French 6 (with momentum) | 0.39% | 2.79 |
| q5 | 0.19% | 1.42 |

Before costs the q5 alpha is 0.29% (t 2.21).

### As a portfolio

| | Long-short fund (T-bills + spread) | Long only (decile 10) | US market |
|---|---|---|---|
| Return per year | 10.7% | 12.5% | 10.1% |
| Volatility per year | 11.4% | 15.8% | 15.3% |
| Sharpe | 0.5 | 0.5 | 0.4 |
| Max drawdown | -30% | -45% | -50% |
| Worst year | -16% (1999) | -25% | -37% (2008) |

The US market is the CRSP value-weighted index (SPY exists only from 1993).
Correlation of the long-short fund with the market: -0.30.

It pays in crises: 1973 +25% and 1974 +29% (market -19% and -28%), 2001 +32% (market -11%),
2008 +63% (market -37%). It lags in speculative rallies: 1999 -16% (market +25%).
The market plus the long-short spread (an overlay, 1:1) has a Sharpe of 0.75 against 0.39,
a worst year of -24% against -37%.

### Is it luck? Does it come from one year?

- Block bootstrap (10,000 resamples, 12-month blocks): no resample with a mean of zero or
  less; 95% interval of the monthly net return 0.24% to 0.77%.
- t-stat 3.90, above the Harvey, Liu and Zhu (2016) bar of 3.
- Without 2008: t 3.38. Without the three best years: t 2.75. Dropping any single year,
  the lowest t is 3.38.
- Rolling 10-year windows: positive in 79% of them. The weak decade is the 1990s (about 1%
  per year), when expensive, unprofitable technology stocks kept rising.

### What it is exposed to

FF6 loadings of the net spread: market -0.10, size -0.14, profitability (RMW) +0.19,
investment (CMA) +0.26. It earns +1.5% per month in months when the market falls and -0.2%
when it rises; +1.3% per month in NBER recessions.

### Robustness

| Check | Result |
|---|---|
| Plain (not industry-relative) quality + value | FF5 alpha before costs 0.51% (t 3.68), q5 0.25% (t 1.51): the industry-relative version is stronger (0.55%, t 4.30; q5 0.29%, t 2.21) |
| 49 industries instead of 12 | net FF5 alpha 0.42% (t 3.19) |
| Intangible-adjusted book value (Eisfeldt, Kim and Papanikolaou 2022) | net 0.53% (t 4.03), net FF5 alpha 0.48% (t 3.68) |
| No microcaps (configs/strategies.yaml) | net 0.38% (t 2.91), net FF6 alpha 0.34% (t 2.29) |
| Capacity | net FF5 alpha with t > 2 up to $500m per side on training ($1bn over 1963-2015) |

## 5. Limits

- After costs, the q5 model explains most of the alpha.
- Weak in the 1990s and, without microcaps, weak in 1995-2015 (net 0.23% per month, t 1.07).
- The return leans on crisis years; 2008 alone is about a sixth of the total.
- It needs short selling (borrow costs and availability are not in the cost model) and
  holds about 140 to 360 stocks per side, so it is a strategy for a fund, not a small
  account.

## 6. Holdout 2016-2025

To be added after the single holdout run
(`python scripts/run_signal.py --signal quality_value_industry --holdout`, logged in
`holdout_log.txt`).

## 7. How to reproduce

```
git pull origin roberto
python src/data/clean.py              # once, needs the WRDS files in data/raw/
python scripts/run_roberto.py         # every result above, about 15-30 minutes
open results/charts/performance.html  # the interactive chart against the market
```

`run_roberto.py` runs each config as one block, saves every script's output in
`results/roberto/logs/` and a summary table in `results/roberto/summary_<block>.txt`.
The 49-industry check downloads Ken French's Siccodes49 file once into
`data/external/raw/`.

## 8. Files on this branch

| File | What it does |
|---|---|
| `src/signals/quality_value_industry.py` | **the strategy** |
| `src/signals/value_industry.py`, `cash_op.py` | its two halves (also tested alone) |
| `src/signals/value_composite.py`, `quality_value.py` | the same ideas without industries, for comparison |
| `src/signals/quality_value_industry_ff49.py`, `quality_value_industry_intan.py` | robustness |
| `src/signals/_industry.py`, `_quality.py` | helpers (industries, ranks, universe) |
| `scripts/run_roberto.py` | runs everything |
| `scripts/analyse_strategy.py` | luck, one-year dependence, factors, market + strategy |
| `scripts/run_capacity.py` | costs and alphas at $100m to $2.5bn |
| `scripts/yearly_performance.py`, `scripts/make_charts.py` | year-by-year table and charts against the market |

The data engine, the cost model, the factor file and the intangible-capital helper
(`src/signals/_intangibles.py`) are the team's, unchanged.

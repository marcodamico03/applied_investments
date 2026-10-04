# applied_investments

Run every command from the project folder. Install the packages once with
`pip install -r requirements.txt`.

To run everything below (except the holdout) in one go: `python scripts/run_all.py`
(`--signals momentum` for just some signals, `--skip-clean` to reuse the cleaned data).

To pick any single step from a menu instead of typing the command:
`python scripts/menu.py` (it shows the exact command before running it).

## Once: data
1. Put the WRDS files in `data/raw/` (`crsp_monthly.csv`, `ccm_funda.csv`).
2. `python src/data/clean.py` writes the cleaned tables to `data/interim/`.
3. `python scripts/run_asset_growth.py` must reproduce the professor's numbers
   (slides 18-27). Rerun it after any change to `clean.py`.

The factor file `data/external/factors_monthly.csv` is in git, so the whole team
uses the same vintage. Only rebuild it on purpose (`python src/data/factors.py`).

## For each signal
1. Add `src/signals/<name>.py`: copy `asset_growth.py` (accounting data) or
   `momentum.py` (market data) and change the formula. The rules are in
   `src/signals/__init__.py`.
2. For a published anomaly, compare it with Ken French's portfolios:
   `python scripts/validate_signal.py --signal <name>`
3. Backtest on train + validation:
   `python scripts/run_signal.py --signal <name>`
   This gives portfolio returns, the spread before and after trading costs,
   alphas for CAPM, FF3-FF6 and q4/q5, the GRS test, and the same spread in other
   universes (no microcaps, price above $5, no financials/utilities).
   Settings are in `configs/default.yaml`; results go to `results/signals/<name>/`.
4. Once, at the very end, when the signal and the config are frozen:
   `python scripts/run_signal.py --signal <name> --holdout`
   The holdout (2016-01 on) is fixed in `scripts/run_signal.py`, not in the configs:
   without `--holdout` every period is cut at 2015-12, whatever the config says.
   Each `--holdout` run adds a line to `holdout_log.txt`: commit it.

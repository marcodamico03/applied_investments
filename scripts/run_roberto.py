"""
Reproduce every result of the quality-value-industry strategy (docs/STRATEGY.md).

Run from the project folder, after the team's data step (data/interim/ must exist:
python src/data/clean.py):
    python scripts/run_roberto.py                 # everything (about 15-30 minutes)
    python scripts/run_roberto.py --block main    # only the main universe (all stocks)

What it runs, in this order (each config is a complete block, because run_signal.py
writes results/signals/<signal>/ for the config that ran last):
  Block "main" (configs/default.yaml, all stocks, training 1963-2005, validation 2006-2015)
    1. run_signal.py for value_composite, value_industry, cash_op, quality_value,
       quality_value_industry_ff49, quality_value_industry_intan, quality_value_industry
    2. summary table of all seven (written before the next block overwrites anything)
    3. analyse_strategy.py, run_capacity.py, yearly_performance.py, make_charts.py
  Block "no_microcaps" (configs/strategies.yaml, stocks above the NYSE 20th percentile)
    4. run_signal.py for quality_value_industry, summary, analyse_strategy.py, run_capacity.py
Every script's full output is saved in results/roberto/logs/; the summaries in
results/roberto/summary_<block>.csv and .txt. The holdout (2016-2025) is never run here.
"""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.alpha import load_factors, regress  # noqa: E402

CORE = "quality_value_industry"
BLOCKS = {
    "main": {
        "config": "configs/default.yaml",
        "signals": ["value_composite", "value_industry", "cash_op", "quality_value",
                    "quality_value_industry_ff49", "quality_value_industry_intan", CORE],
        "after": [["analyse_strategy.py"], ["run_capacity.py"], ["yearly_performance.py"], ["make_charts.py"]],
    },
    "no_microcaps": {
        "config": "configs/strategies.yaml",
        "signals": [CORE],
        "after": [["analyse_strategy.py"], ["run_capacity.py"]],
    },
}
OUT = BASE_DIR / "results" / "roberto"
NEEDED = [BASE_DIR / "data" / "interim" / f for f in ("crsp_clean.parquet", "compustat_clean.parquet",
                                                      "spreads_clean.parquet")]


def run(step: str, args: list) -> None:
    log = OUT / "logs" / f"{step}.txt"
    print(f"  {step:55s}", end="", flush=True)
    with log.open("w") as fh:
        res = subprocess.run([sys.executable, *args], cwd=BASE_DIR, stdout=fh, stderr=subprocess.STDOUT)
    print("ok" if res.returncode == 0 else f"FAILED (see {log.relative_to(BASE_DIR)})")
    if res.returncode:
        sys.exit(1)


def summary(block: str, config: str, signals: list) -> pd.DataFrame:
    cfg = yaml.safe_load((BASE_DIR / config).read_text())
    periods = {"train": tuple(cfg["periods"]["train"]), "validation": tuple(cfg["periods"]["validation"]),
               "1963-2015": ("1963-07", "2015-12")}
    f = load_factors(BASE_DIR / cfg["data"]["factors"])
    rows = []
    for s in signals:
        m = importlib.import_module(f"src.signals.{s}")
        r = pd.read_csv(BASE_DIR / "results" / "signals" / s / "returns_vw.csv", index_col=0)
        r.index = pd.PeriodIndex(r.index, freq="M")
        net = r[f"{m.LONG}-{m.SHORT} net"]
        row = {"signal": s}
        for p, (a, b) in periods.items():
            x = net.loc[a:b].dropna()
            row[f"{p} net %"] = x.mean() * 100
            row[f"{p} t"] = x.mean() / x.std() * np.sqrt(len(x))
            row[f"{p} Sharpe"] = x.mean() / x.std() * np.sqrt(12)
        x = net.loc["1963-07":"2015-12"]
        for model in ("CAPM", "FF5", "FF6", "q5"):
            res = regress(x, f, model, long_short=True, nw_lags=6)
            row[f"{model} net alpha %"], row[f"{model} t"] = res["alpha %"], res["t(alpha)"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("signal")
    table.to_csv(OUT / f"summary_{block}.csv")
    text = table.T.to_string(float_format=lambda v: f"{v:.2f}")
    (OUT / f"summary_{block}.txt").write_text(
        f"Value-weighted long-short spreads after costs, {config}\n\n{text}\n")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", choices=["all", *BLOCKS], default="all")
    args = ap.parse_args()
    missing = [str(p.relative_to(BASE_DIR)) for p in NEEDED if not p.exists()]
    if missing:
        sys.exit(f"Missing {missing}: run the team's data step first (python src/data/clean.py).")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 200)
    for block, spec in BLOCKS.items():
        if args.block not in ("all", block):
            continue
        cfg = spec["config"]
        print(f"\nBlock '{block}' ({cfg})")
        for s in spec["signals"]:
            run(f"{block}_signal_{s}", ["scripts/run_signal.py", "--signal", s, "--config", cfg])
        table = summary(block, cfg, spec["signals"])
        for cmd in spec["after"]:
            extra = [] if cfg == "configs/default.yaml" or cmd[0] == "yearly_performance.py" else ["--config", cfg]
            run(f"{block}_{cmd[0][:-3]}", [f"scripts/{cmd[0]}", *extra])
        print(f"\nSummary, {block}:")
        print(table.T.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nAll outputs in {OUT.relative_to(BASE_DIR)}/ ; charts in results/charts/ (main block).")


if __name__ == "__main__":
    main()

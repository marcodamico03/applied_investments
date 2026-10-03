"""
Run the whole pipeline in one go:
  1. clean the WRDS data (only if data/interim/ is missing it, or with --clean)
  2. reproduce the professor's asset growth numbers (regression test)
  3. for every signal in src/signals/: compare with Ken French (if it has a
     FRENCH_BENCHMARK), then backtest on train + validation

Run from the project folder:
    python scripts/run_all.py
    python scripts/run_all.py --signals momentum book_to_market
    python scripts/run_all.py --clean --config configs/my_test.yaml

Never touches the test period: there is no --holdout here on purpose. Run
    python scripts/run_signal.py --signal <name> --holdout
by hand, once, when the signal and the config are frozen.

Each step runs as its own process. If one fails, the others still run, and the
summary at the end lists what failed. The full output of every step is saved
in results/logs/<step>.log.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # lets Python find the "src" folder

from src.signals import available, load_signal  # noqa: E402

INTERIM_DIR = BASE_DIR / "data" / "interim"
CLEAN_FILES = ["crsp_clean.parquet", "compustat_clean.parquet"]
LOG_DIR = BASE_DIR / "results" / "logs"


def run_step(label: str, args: list) -> tuple:
    """Run one script, show its output live and save it to a log file.
    Returns (ok, seconds)."""
    print(f"\n{'=' * 78}\n>> {label}\n{'=' * 78}", flush=True)
    log_file = LOG_DIR / f"{label.replace(' ', '_').replace(':', '')}.log"
    start = time.time()
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-u", *args], cwd=BASE_DIR, text=True, encoding="utf-8",
            errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()
    seconds = time.time() - start
    ok = proc.returncode == 0
    if not ok:
        print(f"!! {label} failed (exit code {proc.returncode}), see {log_file}")
    return ok, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole pipeline.")
    parser.add_argument("--signals", nargs="+", default=None,
                        help=f"signals to run (default: all of {', '.join(available())})")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--clean", action="store_true", help="rerun clean.py even if its output exists")
    parser.add_argument("--skip-validation", action="store_true", help="skip the Ken French comparisons")
    parser.add_argument("--skip-regression", action="store_true", help="skip run_asset_growth.py")
    args = parser.parse_args()

    signals = args.signals or available()
    for name in signals:
        load_signal(name)  # fails now, not halfway through, if a name is wrong

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    steps = []  # (label, script arguments)
    if args.clean or not all((INTERIM_DIR / f).exists() for f in CLEAN_FILES):
        steps.append(("clean data", ["src/data/clean.py"]))
    if not args.skip_regression:
        steps.append(("regression asset_growth", ["scripts/run_asset_growth.py"]))
    for name in signals:
        if not args.skip_validation and getattr(load_signal(name), "FRENCH_BENCHMARK", None):
            steps.append((f"validate {name}", ["scripts/validate_signal.py", "--signal", name]))
        steps.append((f"backtest {name}",
                      ["scripts/run_signal.py", "--signal", name, "--config", args.config]))

    print(f"{len(steps)} steps: " + ", ".join(label for label, _ in steps))
    results = []
    for label, script_args in steps:
        ok, seconds = run_step(label, script_args)
        results.append((label, ok, seconds))
        if label == "clean data" and not ok:
            print("!! Cleaning failed: everything after it needs the clean data, stopping.")
            break

    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    for label, ok, seconds in results:
        print(f"  {'OK    ' if ok else 'FAILED'}  {label:<35} {seconds / 60:5.1f} min")
    print(f"Logs in: {LOG_DIR}")
    if not all(ok for _, ok, _ in results) or len(results) < len(steps):
        sys.exit(1)


if __name__ == "__main__":
    main()

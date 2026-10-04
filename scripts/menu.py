"""
Interactive menu: choose a script of the project, answer a few questions (signal,
config, options) and it runs. Before running, the menu prints the exact command,
so you can also copy it and run it yourself next time.

Run from the project folder:
    python scripts/menu.py

At every question: Enter takes the value in [brackets], b goes back to the menu,
q quits. Ctrl+C stops a running script and brings you back to the menu.
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # lets Python find the "src" folder

from src.signals import available, load_signal  # noqa: E402

CONFIG_DIR = BASE_DIR / "configs"
CLEAN_FILES = [BASE_DIR / "data" / "interim" / f for f in ("crsp_clean.parquet", "compustat_clean.parquet")]
HOLDOUT_LOG = BASE_DIR / "holdout_log.txt"


class Back(Exception):
    """The user typed b: back to the main menu."""


class Quit(Exception):
    """The user typed q."""


# ------------------------------------------------------------------------------
# Questions
# ------------------------------------------------------------------------------
def ask(prompt: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    answer = input(f"{prompt}{shown}: ").strip()
    if answer.lower() == "b":
        raise Back
    if answer.lower() == "q":
        raise Quit
    return answer or default


def show_list(title: str, options: List[str]) -> None:
    print(f"\n{title}")
    for i, label in enumerate(options, 1):
        print(f"  {i:>2}. {label}")


def pick_one(title: str, options: List[str], default: Optional[int] = None) -> int:
    """Numbered list; returns the position (from 0) of the option chosen."""
    show_list(title, options)
    while True:
        answer = ask("Number", str(default + 1) if default is not None else "")
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print(f"  Type a number from 1 to {len(options)} (b = back, q = quit).")


def pick_many(title: str, options: List[str]) -> List[int]:
    """Numbered list; several numbers separated by spaces, Enter = all."""
    show_list(title, options)
    while True:
        parts = ask("Numbers separated by spaces (Enter = all)").replace(",", " ").split()
        if not parts:
            return list(range(len(options)))
        if all(p.isdigit() and 1 <= int(p) <= len(options) for p in parts):
            return sorted({int(p) - 1 for p in parts})
        print(f"  Type numbers from 1 to {len(options)}, e.g. 1 3 4 (b = back, q = quit).")


def yes(prompt: str, default: bool) -> bool:
    return ask(f"{prompt} (y/n)", "y" if default else "n").lower().startswith("y")


def shorten(text: str, width: int = 58) -> str:
    return text if len(text) <= width else text[:width - 3] + "..."


def describe_signal(name: str) -> str:
    """'momentum          monthly  Momentum signal (Jegadeesh and Titman 1993), ...'"""
    spec = load_signal(name)
    lines = (spec.__doc__ or "").strip().splitlines()
    return f"{name:<17} {spec.REBALANCE:<8} {shorten(lines[0] if lines else '')}"


def pick_signal(names: Optional[List[str]] = None) -> str:
    names = names or available()
    return names[pick_one("Which signal?", [describe_signal(n) for n in names])]


def pick_config() -> str:
    """configs/<file>.yaml, default.yaml first. Each one is shown with its first comment line."""
    files = sorted(CONFIG_DIR.glob("*.yaml"), key=lambda p: (p.name != "default.yaml", p.name))
    labels = []
    for path in files:
        comments = [l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("#")]
        about = comments[0].lstrip("# ").strip() if comments else ""
        labels.append(f"{path.name:<17} {shorten(about)}")
    return f"configs/{files[pick_one('Which config?', labels, default=0)].name}"


# ------------------------------------------------------------------------------
# Menu entries: each one asks its questions and returns the script and arguments
# ------------------------------------------------------------------------------
def clean_data() -> List[str]:
    print("\nRebuilds data/interim/ from the WRDS files in data/raw/ (a few minutes).")
    print("Afterwards, rerun the asset growth check (it must still match the professor).")
    return ["src/data/clean.py"]


def asset_growth_check() -> List[str]:
    return ["scripts/run_asset_growth.py"]


def asset_growth_alphas() -> List[str]:
    if not (BASE_DIR / "results" / "asset_growth" / "ew_returns.csv").exists():
        print("\n!! This needs results/asset_growth/: run the asset growth check first.")
        raise Back
    return ["scripts/run_alphas_asset_growth.py"]


def validate() -> List[str]:
    names = [n for n in available() if getattr(load_signal(n), "FRENCH_BENCHMARK", None)]
    return ["scripts/validate_signal.py", "--signal", pick_signal(names)]


def backtest() -> List[str]:
    signal = pick_signal()
    return ["scripts/run_signal.py", "--signal", signal, "--config", pick_config()]


def pipeline() -> List[str]:
    mode = pick_one("Which signals?", [
        "All signals",
        "Only the final strategies (--final: configs/strategies.yaml, no French or asset growth check)",
        "Choose the signals",
    ])
    args = ["scripts/run_all.py"]
    if mode == 1:
        args.append("--final")
    else:
        if mode == 2:
            names = available()
            chosen = pick_many("Which signals?", [describe_signal(n) for n in names])
            args += ["--signals"] + [names[i] for i in chosen]
        args += ["--config", pick_config()]
        if not yes("Compare with Ken French (signals that have a benchmark)?", True):
            args.append("--skip-validation")
        if not yes("Run the asset growth check against the professor's numbers?", True):
            args.append("--skip-regression")
    if all(f.exists() for f in CLEAN_FILES):
        if not yes("Clean the WRDS data again first? Slow, only needed after a change "
                   "to clean.py or data/raw", False):
            args.append("--skip-clean")
    else:
        print("\nNo cleaned data in data/interim/ yet: the pipeline starts by cleaning it.")
    return args


def holdout() -> List[str]:
    print("\n!! The holdout (2016-01 on) is used ONCE per signal, when the signal and the")
    print("!! config are frozen. Every run adds a line to holdout_log.txt: commit it.")
    print("!! Signals run with configs/strategies.yaml before 2026-10-04 already saw the")
    print("!! holdout years without a line in the log.")
    signal = pick_signal()
    lines = HOLDOUT_LOG.read_text().splitlines() if HOLDOUT_LOG.exists() else []
    previous = [line for line in lines if f"signal={signal}" in line.split()]
    if previous:
        print(f"\n!! The holdout of '{signal}' was already used {len(previous)} time(s):")
        for line in previous:
            print(f"!!   {line}")
    config = pick_config()
    if ask(f"Type the signal name ({signal}) to confirm") != signal:
        print("Not confirmed: nothing was run.")
        raise Back
    return ["scripts/run_signal.py", "--signal", signal, "--config", config, "--holdout"]


def factors() -> List[str]:
    choice = pick_one("The factor file is the team's shared vintage (in git). What do you want?", [
        "Add or refresh only the short-term reversal column STREV (--add-reversal)",
        "Rebuild the whole file from Ken French and global-q (past values change!)",
    ])
    if choice == 0:
        return ["src/data/factors.py", "--add-reversal"]
    print("\n!! A rebuild downloads the latest vintages: past factor values change a little,")
    print("!! so alphas computed before are no longer comparable. Commit the new file for the team.")
    if ask("Type rebuild to confirm") != "rebuild":
        print("Not confirmed: nothing was run.")
        raise Back
    return ["src/data/factors.py"]


def quit_menu() -> List[str]:
    raise Quit


# In the order of the README: data, checks, signals, holdout, factor file
MENU = [
    ("Clean the WRDS data", "clean.py", clean_data),
    ("Check asset growth vs the professor", "run_asset_growth.py", asset_growth_check),
    ("Asset growth alphas", "run_alphas_asset_growth.py", asset_growth_alphas),
    ("Compare a signal with Ken French", "validate_signal.py", validate),
    ("Backtest a signal", "run_signal.py", backtest),
    ("Run the whole pipeline", "run_all.py", pipeline),
    ("Holdout run of a signal (ONCE)", "run_signal.py --holdout", holdout),
    ("Update the factor file", "factors.py", factors),
    ("Quit", "", quit_menu),
]


def run(args: List[str]) -> None:
    print(f"\nCommand:  python {' '.join(args)}")
    if not yes("Run it?", True):
        return
    try:
        code = subprocess.call([sys.executable, *args], cwd=BASE_DIR)
    except KeyboardInterrupt:
        print("\n!! Stopped (Ctrl+C).")
        return
    print("\nFinished." if code == 0 else f"\n!! Failed (exit code {code}): see the messages above.")


def main() -> None:
    print("Applied Investments: choose what to run (b = back, q = quit).")
    labels = [f"{label:<37} ({script})" if script else label for label, script, _ in MENU]
    while True:
        try:
            run(MENU[pick_one("What do you want to run?", labels)][2]())
        except Back:
            continue
        except (Quit, EOFError, KeyboardInterrupt):
            print("\nBye.")
            return


if __name__ == "__main__":
    main()
